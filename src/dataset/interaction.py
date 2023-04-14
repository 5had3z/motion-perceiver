"""Dataset for the INTERACTION dataset that consists of folders of CSVs
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from multiprocessing import get_context, Queue
from pathlib import Path
from subprocess import run
from typing import Any, Dict, List, Tuple
from warnings import warn
from xml.etree import ElementTree

from tqdm.auto import tqdm
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    warn("Cannot import tensorflow, but only required for data export")

from konductor.modules.data import DATASET_REGISTRY, DatasetConfig, Mode

from nvidia.dali import pipeline_def, Pipeline
from nvidia.dali.types import DALIDataType
import nvidia.dali.math as dmath
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec

_MAX_AGENTS: int = 64
_MAX_ROADGRAPH: int = 1024
_TIMESPAN: int = 40


def _cache_record_idx(dataset_path: Path) -> Path:
    """
    Initially try to make with tf record dali index
    in folder adjacent to dataset suffixed by idx.
    If that fails due to permission requirements, make in /tmp.
    """
    dali_idx_path = dataset_path.parent / f"{dataset_path.name}_dali_idx"
    if not dali_idx_path.exists():
        try:
            dali_idx_path.mkdir()
            return dali_idx_path
        except OSError:
            print(
                f"Unable to create dali index at {dali_idx_path},"
                f" changing to /tmp/{dataset_path.name}_dali_idx"
            )

            dali_idx_path = Path(f"/tmp/{dataset_path.name}_dali_idx")
            if not dali_idx_path.exists():
                dali_idx_path.mkdir()

    return dali_idx_path


@pipeline_def
def interation_pipeline(
    record_file: Path,
    shard_id: int,
    num_shards: int,
    random_shuffle: bool,
    full_sequence: bool = False,
    vehicle_features: List[str] = None,
    road_features: bool = False,
    roadmap: bool = False,
    signal_features: bool = False,
    map_normalize: float = 0.0,
    occupancy_size: int = 0,
    heatmap_time: List[int] = None,
    filter_future: bool = True,
    separate_classes: bool = False,
    random_heatmap_minmax: Tuple[int, int] = None,
    random_heatmap_count: int = 0,
):
    # fmt: off
    # Vehicle_features is order sensitive (this is ordering of channel concatenation)
    if vehicle_features is None:
        vehicle_features = [
            "state/x", "state/y", "state/t",
            "state/vx", "state/vy", "state/vt",
            "state/width", "state/length"
        ]

    # Features of the road.
    roadgraph_features = {
        "roadgraph/id": tfrec.FixedLenFeature([_MAX_ROADGRAPH, 1], tfrec.int64, 0),
        "roadgraph/type": tfrec.FixedLenFeature([_MAX_ROADGRAPH, 1], tfrec.int64, 0),
        "roadgraph/valid": tfrec.FixedLenFeature([_MAX_ROADGRAPH, 1], tfrec.int64, 0),
        "roadgraph/xyz": tfrec.FixedLenFeature([_MAX_ROADGRAPH, 3], tfrec.float32, 0.0),
    }
    
    # Features of other agents.
    state_features = {
        "state/x": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/y": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/t": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/vx": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/vy": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/length": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/width": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.float32, 0.0),
        "state/timestamp_ms": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.int64, 0),
        "state/valid": tfrec.FixedLenFeature([_MAX_AGENTS, _TIMESPAN], tfrec.int64, 0),
        "state/id": tfrec.FixedLenFeature([_MAX_AGENTS, 1], tfrec.int64, 0),
        "state/type": tfrec.FixedLenFeature([_MAX_AGENTS], tfrec.int64, 0),
    }
    # fmt: on

    features_description = {}
    features_description.update(roadgraph_features)
    features_description.update(state_features)

    tfrec_idx_root = _cache_record_idx(record_file.parent)
    tfrec_idx = tfrec_idx_root / f"{record_file.name}.idx"

    if not tfrec_idx.exists():
        run(["tfrecord2idx", str(record_file), str(tfrec_idx)])

    inputs = fn.readers.tfrecord(
        path=str(record_file),
        index_path=str(tfrec_idx),
        features=features_description,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=random_shuffle,
        name=f"{record_file.stem}",
    )

    inputs["state/valid"] = fn.cast(inputs[f"state/valid"], dtype=DALIDataType.INT32)

    # Center coordinate system based off vehicles
    center_x = fn.masked_median(inputs["state/x"], inputs["state/valid"])
    center_y = fn.masked_median(inputs["state/y"], inputs["state/valid"])

    inputs["state/x"] = inputs["state/x"] - center_x
    inputs["state/y"] = inputs["state/y"] - center_y

    # Center the map at 0,0 and divide by normalization factor
    if map_normalize > 0.0:
        for key in ["state/x", "state/y", "state/width", "state/length"]:
            inputs[key] = inputs[key] / map_normalize

    if "state/t" in inputs:  # normalize angle bewteen -/+ pi
        if "state/vt" in vehicle_features:
            # will be chat during theta wrapping
            inputs["state/vt"] = inputs["state/t"][:, 1:] - inputs["state/t"][:, :-1]
            # concat same yaw rate to the end and assume no jerk (ceebs using at)
            inputs["state/vt"] = fn.cat(
                inputs["state/vt"], inputs["state/vt"][:, -1:], axis=1
            )
        inputs["state/t"] = dmath.atan2(
            dmath.sin(inputs["state/t"]), dmath.cos(inputs["state/t"])
        )

    inputs["state/type"] = fn.cast(inputs["state/type"], dtype=DALIDataType.FLOAT)
    inputs["state/class"] = fn.stack(
        *[inputs["state/type"] for _ in range(_TIMESPAN if full_sequence else 1)],
        axis=1,
    )

    if separate_classes:
        vehicle_features.append("state/class")

    outputs = [
        fn.stack(*[inputs[k] for k in vehicle_features], axis=2),
        inputs["state/valid"],
    ]

    if roadmap:
        outputs.append(
            fn.roadgraph_image(
                inputs["roadgraph/xyz"],
                inputs["roadgraph/type"],
                inputs["roadgraph/id"],
                inputs["roadgraph/valid"],
                center_x,
                center_y,
                size=occupancy_size,
                normalize_value=map_normalize,
                # lane_markings=True,
                lane_center=True,
            )
        )

    # Add occupancy heatmap
    if occupancy_size > 0:
        if heatmap_time is None:
            heatmap_time = [10] if full_sequence else [0]

        occ_kwargs = {
            "size": occupancy_size,
            "const_time_idx": heatmap_time,
            "filter_future": filter_future,
            "separate_classes": separate_classes,
        }

        if random_heatmap_count > 0:
            occ_kwargs["n_random_idx"] = random_heatmap_count
            occ_kwargs["min_random_idx"] = random_heatmap_count[0]
            occ_kwargs["max_random_idx"] = random_heatmap_count[1]

        outputs.extend(
            fn.occupancy_mask(
                inputs["state/x"],
                inputs["state/y"],
                inputs["state/t"],
                inputs["state/width"],
                inputs["state/length"],
                inputs["state/valid"],
                inputs["state/class"],
                **occ_kwargs,
            )
        )

    return tuple([o.gpu() for o in outputs])


@dataclass
@DATASET_REGISTRY.register_module("interacton")
class InteractionConfig(DatasetConfig):
    full_sequence: bool = False
    vehicle_features: List[str] | None = None
    road_features: bool = False
    roadmap: bool = False
    signal_features: bool = False
    map_normalize: float = 0.0
    occupancy_size: int = 0
    heatmap_time: List[int] | None = None
    filter_future: bool = True
    separate_classes: bool = False
    random_heatmap_minmax: Tuple[int, int] | None = None
    random_heatmap_count: int = 0
    occupancy_roi: float = 1.0

    @property
    def properties(self) -> Dict[str, Any]:
        return asdict(self)

    def get_instance(self, mode: Mode, **kwargs) -> Tuple[Pipeline, List[str], str]:
        root = {
            Mode.train: self.basepath / "training",
            Mode.val: self.basepath / "validation",
            Mode.test: self.basepath / "testing",
        }[mode]

        pipe_kwargs = asdict(self)
        del pipe_kwargs["basepath"]

        output_map = ["agents", "agents_valid"]
        if self.road_features:
            output_map.extend(["roadgraph", "roadgraph_valid"])
        if self.roadmap:
            output_map.append("roadmap")
        if self.signal_features:
            output_map.extend(["signals", "signals_valid"])
        if self.occupancy_size > 0:
            output_map.extend(["heatmap", "time_idx"])

        return interation_pipeline(root, **pipe_kwargs, **kwargs), output_map, root.stem


class IClass(Enum):
    CAR = auto()
    HUMAN = auto()
    BICYCLE = auto()


@dataclass
class InteractionSample:
    case: int
    track: int
    frame: int
    ts_ms: int
    agent: IClass
    x: float
    y: float
    vx: float
    vy: float
    theta: float
    length: float
    width: float

    @classmethod
    def from_csv_line(cls, data: str) -> InteractionSample:
        """Create from a line of csv"""
        pargs = []
        for idx, sample in enumerate(data.split(",")):
            if idx < 4:  # case, track, frame, ts are int
                try:
                    pargs.append(int(sample))
                except ValueError:
                    # sometimes case_id is float for some reason
                    pargs.append(int(float(sample)))
            elif idx == 4:  # convert class string to enum
                if sample == "car":
                    pargs.append(IClass.CAR)
                elif "bicycle" in sample:
                    pargs.append(IClass.BICYCLE)
                else:
                    raise KeyError(f"Unidentified class {sample}")
            else:  # x,y,vx,vy,psi_rad,l,w
                if sample != "":
                    pargs.append(float(sample))
                else:  # l,w of pedestrian types are empty, add placeholders
                    pargs.append(1.0)

        return cls(*pargs)

    def time_index(self) -> int:
        """Returns index of time sequence based of timestamp"""
        return self.ts_ms // 100 - 1


@dataclass
class InteractionRecord:
    scenario_id: int

    # fmt: off
    state_x: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, _TIMESPAN), dtype=float))
    state_y: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, _TIMESPAN), dtype=float))
    state_t: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, _TIMESPAN), dtype=float))
    state_vx: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, _TIMESPAN), dtype=float))
    state_vy: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, _TIMESPAN), dtype=float))
    state_length: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, _TIMESPAN), dtype=float))
    state_width: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, _TIMESPAN), dtype=float))
    state_timestamp: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, _TIMESPAN), dtype=int))
    state_valid: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, _TIMESPAN), dtype=int))
    state_type: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, 1), dtype=int))
    state_id: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_AGENTS, 1), dtype=int))

    roadgraph_xyz: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_ROADGRAPH, 3), dtype=int))
    roadgraph_valid: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_ROADGRAPH, 1), dtype=int))
    roadgraph_id: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_ROADGRAPH, 1), dtype=int))
    roadgraph_type: np.ndarray = field(default_factory=lambda: np.zeros((_MAX_ROADGRAPH, 1), dtype=int))
    # fmt: on

    def add_sample(self, sample: InteractionSample) -> None:
        """Adds interaction sample to record"""
        self.state_x[sample.track, sample.time_index()] = sample.x
        self.state_y[sample.track, sample.time_index()] = sample.y
        self.state_t[sample.track, sample.time_index()] = sample.theta
        self.state_vx[sample.track, sample.time_index()] = sample.vx
        self.state_vy[sample.track, sample.time_index()] = sample.vy
        self.state_length[sample.track, sample.time_index()] = sample.length
        self.state_width[sample.track, sample.time_index()] = sample.width
        self.state_timestamp[sample.track, sample.time_index()] = sample.ts_ms
        self.state_valid[sample.track, sample.time_index()] = 1
        self.state_id[sample.track] = sample.track
        self.state_type[sample.track] = sample.agent.value

    def add_roadgraph_tf(self, roadmap: InteractionMap) -> None:
        """Add roadgraph already formatted as tf vectors"""
        roadgraph = roadmap.tf_serialise()
        self.roadgraph_id = roadgraph["id"]
        self.roadgraph_type = roadgraph["type"]
        self.roadgraph_valid = roadgraph["valid"]
        self.roadgraph_xyz = roadgraph["xyz"]

    def tf_serialise(self) -> tf.train.Example:
        """Serialise"""
        tf_sample = tf.train.Example(
            features=tf.train.Features(
                # fmt: off
                feature={
                    "state/x":              tf.train.Feature(float_list=tf.train.FloatList(value=self.state_x.flatten())),
                    "state/y":              tf.train.Feature(float_list=tf.train.FloatList(value=self.state_y.flatten())),
                    "state/t":              tf.train.Feature(float_list=tf.train.FloatList(value=self.state_t.flatten())),
                    "state/vx":             tf.train.Feature(float_list=tf.train.FloatList(value=self.state_vx.flatten())),
                    "state/vy":             tf.train.Feature(float_list=tf.train.FloatList(value=self.state_vy.flatten())),
                    "state/length":         tf.train.Feature(float_list=tf.train.FloatList(value=self.state_length.flatten())),
                    "state/width":          tf.train.Feature(float_list=tf.train.FloatList(value=self.state_width.flatten())),
                    "state/timestamp_ms":   tf.train.Feature(int64_list=tf.train.Int64List(value=self.state_timestamp.flatten())),
                    "state/valid":          tf.train.Feature(int64_list=tf.train.Int64List(value=self.state_valid.flatten())),
                    "state/id":             tf.train.Feature(int64_list=tf.train.Int64List(value=self.state_id.flatten())),
                    "state/type":           tf.train.Feature(int64_list=tf.train.Int64List(value=self.state_type.flatten())),
                    "roadgraph/xyz":         tf.train.Feature(float_list=tf.train.FloatList(value=self.roadgraph_xyz.flatten())),
                    "roadgraph/valid":      tf.train.Feature(int64_list=tf.train.Int64List(value=self.roadgraph_valid.flatten())),
                    "roadgraph/id":         tf.train.Feature(int64_list=tf.train.Int64List(value=self.roadgraph_id.flatten())),
                    "roadgraph/type":       tf.train.Feature(int64_list=tf.train.Int64List(value=self.roadgraph_type.flatten())),
                }
                # fmt: on
            )
        )
        return tf_sample.SerializeToString()


class SplineType(Enum):
    LANECENTER_FREEWAY = 1
    LANECENTER_STREET = 2
    LANECENTER_BIKE = 3
    ROADLINE_BROKENSINGLEWHITE = 6
    ROADLINE_SOLIDSINGLEWHITE = 7
    ROADLINE_SOLIDDOUBLEWHITE = 8
    ROADLINE_BROKENSINGLEYELLOW = 9
    ROADLINE_BROKENDOUBLEYELLOW = 10
    ROADLINE_SOLIDSINGLEYELLOW = 11
    ROADLINE_SOLIDDOUBLEYELLOW = 12
    ROADLINE_PASSINGDOUBLEYELLOW = 13
    ROADEDGE_BOUNDARY = 15
    ROADEDGE_MEDIAN = 16
    STOPSIGN = 17
    CROSSWALK = 18
    SPEEDBUMP = 19

    @classmethod
    def from_xml(cls, xml_elem: ElementTree.ElementTree):
        """Converts xml element to waymo enum"""
        type_ = xml_elem.find('.//tag[@k="type"]').attrib["v"]

        if type_ in ["line_thin", "line_thick"]:
            enumstr = "ROADLINE_"
            subtype_ = xml_elem.find('.//tag[@k="subtype"]').attrib["v"]

            if subtype_ == "solid":
                enumstr += "SOLIDSINGLE"
            elif subtype_ == "dashed":
                enumstr += "BROKENSINGLE"
            elif subtype_ == "solid_solid":
                enumstr += "SOLIDDOUBLE"
            else:
                raise AttributeError(f"Unidentified subtype {subtype_}")

            color_ = xml_elem.find('.//tag[@k="color"]')
            if color_ is None or color_.attrib["v"] == "white":
                enumstr += "WHITE"
            elif color_.attrib["v"] == "yellow":
                enumstr += "YELLOW"
            else:
                raise AttributeError(f"Unidentified color {color_.attrib['v']}")

            return cls[enumstr]

        elif type_ == "pedestrian_marking":
            return cls["CROSSWALK"]

        elif type_ == "virtual":
            return cls["LANECENTER_STREET"]

        elif type_ in ["road_border", "curbstone"]:
            return cls["ROADEDGE_BOUNDARY"]

        elif type_ in [
            "traffic_sign",
            "stop_line",
            "guard_rail",
        ]:
            # Skipping traffic_signas I'm not sure how to handle it since
            # the x,y,t of the sign isn't given, only its relation to lanes
            return None
        else:
            raise AttributeError(f"Unidentified type {type_}")


@dataclass
class InteractionSpline:
    id: int
    type: SplineType
    points: np.ndarray


@dataclass
class InteractionMap:
    name: str
    splines: List[InteractionSpline] = field(default_factory=list)

    def tf_serialise(self) -> Dict[str, np.ndarray]:
        """
        Returns serialised version of the map
        required for dali loader {id, type, valid, xyz}
        """
        xyz = np.zeros((_MAX_ROADGRAPH, 3), dtype=int)
        valid = np.zeros((_MAX_ROADGRAPH, 1), dtype=int)
        id = np.zeros((_MAX_ROADGRAPH, 1), dtype=int)
        stype = np.zeros((_MAX_ROADGRAPH, 1), dtype=int)

        start_idx = 0
        for spline in self.splines:
            end_idx = start_idx + spline.points.shape[0]
            assert (
                end_idx < _MAX_ROADGRAPH
            ), f"MAX_ROADGRAPH needs to be increased, {end_idx=}>{_MAX_ROADGRAPH}"
            xyz[start_idx:end_idx, 0:2] = spline.points
            valid[start_idx:end_idx] = 1
            id[start_idx:end_idx] = spline.id
            stype[start_idx:end_idx] = spline.type.value
            start_idx = end_idx

        return {"xyz": xyz, "valid": valid, "id": id, "type": stype}


def construct_spline(
    osm_data: ElementTree.ElementTree, way: ElementTree.Element
) -> InteractionSpline:
    """Find the node data corresponding to the way and construct a spline description"""
    spline_type = SplineType.from_xml(way)
    if spline_type is None:
        return None

    points = []
    for node in way.iter("nd"):
        node_id = node.attrib["ref"]
        node_data = osm_data.find(f'.//node[@id="{node_id}"]')
        points.append([float(node_data.attrib["x"]), float(node_data.attrib["y"])])

    return InteractionSpline(
        id=int(way.attrib["id"]), type=spline_type, points=np.array(points)
    )


def parse_map_osm(osm_path: Path) -> InteractionMap:
    """Parse OSM file into InteractionMap"""
    print(f"Parsing map: {osm_path}")
    with open(osm_path, "r", encoding="utf-8") as osm_f:
        osm_data = ElementTree.parse(osm_f)

    imap = InteractionMap(osm_path.stem)
    for way in osm_data.findall("way"):
        spline = construct_spline(osm_data, way)
        if spline is not None:
            imap.splines.append(spline)

    print(f"Map Done: {osm_path}")
    return imap


def get_map_path(csv_file: Path) -> Path:
    """Determine the map file path corresponding to the data csv"""
    base_path = csv_file.parent.parent
    filename = "_".join(csv_file.stem.split("_")[:-1])  # remove _train / _val suffix
    return base_path / "maps" / f"{filename}.osm_xy"


def parse_csv_to_datasets(
    file_idx: int, csv_file: Path, dataq: Queue = None
) -> List[InteractionRecord]:
    """"""
    tf_datasets: Dict[int, InteractionRecord] = {}
    print(f"Started [{file_idx}]: {csv_file.name}")

    imap = parse_map_osm(get_map_path(csv_file))

    with open(csv_file, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        while line := f.readline().strip():
            sample = InteractionSample.from_csv_line(line)
            sample.case = 100 * sample.case + file_idx
            if sample.case not in tf_datasets:
                tf_datasets[sample.case] = InteractionRecord(sample.case)
            tf_datasets[sample.case].add_sample(sample)

    for dataset in tf_datasets:
        tf_datasets[dataset].add_roadgraph_tf(imap)

    print(f"Finished Parsing [{file_idx}]: {csv_file.name}")

    if dataq is not None:
        for data in tf_datasets:
            dataq.put(data)
    else:
        return list(tf_datasets.values())


def split_to_record(split_folder: Path, output_rec: Path) -> None:
    """Converts training/validation split csvs to singular TFRecord Dataset"""
    print(f"Processing Dataset: {split_folder.name}")

    # Generate intermediate dataset format
    # Python multiprocessing takes two decades to gather to root process
    # need to mess around with getting queues to work correctly (iteratively pipe results back)
    # while there are workers, yield from pipe, else finished
    # ctx = get_context("spawn")
    # dataq = ctx.Queue()
    # subset = [[i, f, None] for i, f in enumerate(split_folder.glob("*.csv"))][2:5]
    # with ctx.Pool(processes=8) as mp:
    #     tf_datasets = [mp.apply_async(parse_csv_to_datasets, s) for s in subset]
    #     mp.close()
    #     mp.join()
    #     for i, waiter in enumerate(tf_datasets):
    #         waiter.ready()
    #         # dataq.get()
    #         waiter.get()
    #         print(f"{i} recieved")

    # # Flatten to 1D List
    # tf_datasets = functools.reduce(operator.iconcat, tf_datasets, [])
    # print(f"{split_folder.name}: flattened datasets")

    tf_datasets = []
    for fidx, csv_file in enumerate(split_folder.glob("*.csv")):
        tf_datasets.extend(parse_csv_to_datasets(fidx, csv_file))

    # Write Dataset to TF Record
    with tf.io.TFRecordWriter(str(output_rec)) as filew:
        with tqdm(
            total=len(tf_datasets), desc=f"Serialising {split_folder.name} tfrecord"
        ) as pbar:
            for tf_dataset in tf_datasets:
                filew.write(tf_dataset.tf_serialise())
                pbar.update(1)


def run_mp(data_root: Path, output_path: Path) -> None:
    """"""
    mp = get_context("spawn").Pool(processes=2)
    for split in ["train", "val"]:
        split_folder = data_root / split
        record_path = output_path / f"interaction_{split}.tfrecord"
        mp.apply_async(split_to_record, [split_folder, record_path])
    mp.close()
    mp.join()


def run_serial(data_root: Path, output_path: Path) -> None:
    """"""
    for split in ["val", "train"]:
        split_folder = data_root / split
        record_path = output_path / f"interaction_{split}.tfrecord"
        split_to_record(split_folder, record_path)


def convert_dataset() -> None:
    """Converts fragmented CSVs to single TFRecord Database for Training"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", help="Path to top-level directory", type=Path)
    args = parser.parse_args()

    tf_dir: Path = args.root / "tfrecord"
    if not tf_dir.exists():
        tf_dir.mkdir()

    run_mp(args.root, tf_dir)


if __name__ == "__main__":
    convert_dataset()
