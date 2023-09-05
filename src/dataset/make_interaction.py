from dataclasses import dataclass, field
from enum import Enum, auto
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree

import numpy as np
import tensorflow as tf
import typer
from interaction import _MAX_AGENTS, _MAX_ROADGRAPH, _TIMESPAN
from konductor.utilities.pbar import LivePbar
from typing_extensions import Annotated

app = typer.Typer()


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
    def from_csv_line(cls, data: str):
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

        if type_ in {"line_thin", "line_thick"}:
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

        if type_ == "pedestrian_marking":
            return cls["CROSSWALK"]

        if type_ == "virtual":
            return cls["LANECENTER_STREET"]

        if type_ in {"road_border", "curbstone"}:
            return cls["ROADEDGE_BOUNDARY"]

        if type_ in {"traffic_sign", "stop_line", "guard_rail"}:
            # Skipping traffic_signas I'm not sure how to handle it since
            # the x,y,t of the sign isn't given, only its relation to lanes
            return None

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
        uid = np.zeros((_MAX_ROADGRAPH, 1), dtype=int)
        stype = np.zeros((_MAX_ROADGRAPH, 1), dtype=int)

        start_idx = 0
        for spline in self.splines:
            end_idx = start_idx + spline.points.shape[0]
            assert (
                end_idx < _MAX_ROADGRAPH
            ), f"MAX_ROADGRAPH needs to be increased, {end_idx=}>{_MAX_ROADGRAPH}"
            xyz[start_idx:end_idx, 0:2] = spline.points
            valid[start_idx:end_idx] = 1
            uid[start_idx:end_idx] = spline.id
            stype[start_idx:end_idx] = spline.type.value
            start_idx = end_idx

        return {"xyz": xyz, "valid": valid, "id": uid, "type": stype}


@dataclass
class InteractionRecord:
    scenario_id: str

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
                    "state/x":
                        tf.train.Feature(float_list=tf.train.FloatList(value=self.state_x.flatten())),
                    "state/y":
                        tf.train.Feature(float_list=tf.train.FloatList(value=self.state_y.flatten())),
                    "state/t":
                        tf.train.Feature(float_list=tf.train.FloatList(value=self.state_t.flatten())),
                    "state/vx":
                        tf.train.Feature(float_list=tf.train.FloatList(value=self.state_vx.flatten())),
                    "state/vy":
                        tf.train.Feature(float_list=tf.train.FloatList(value=self.state_vy.flatten())),
                    "state/length":
                        tf.train.Feature(float_list=tf.train.FloatList(value=self.state_length.flatten())),
                    "state/width":
                        tf.train.Feature(float_list=tf.train.FloatList(value=self.state_width.flatten())),
                    "state/timestamp_ms":
                        tf.train.Feature(int64_list=tf.train.Int64List(value=self.state_timestamp.flatten())),
                    "state/valid":
                        tf.train.Feature(int64_list=tf.train.Int64List(value=self.state_valid.flatten())),
                    "state/id":
                        tf.train.Feature(int64_list=tf.train.Int64List(value=self.state_id.flatten())),
                    "state/type":
                        tf.train.Feature(int64_list=tf.train.Int64List(value=self.state_type.flatten())),
                    "roadgraph/xyz":
                        tf.train.Feature(float_list=tf.train.FloatList(value=self.roadgraph_xyz.flatten())),
                    "roadgraph/valid":
                        tf.train.Feature(int64_list=tf.train.Int64List(value=self.roadgraph_valid.flatten())),
                    "roadgraph/id":
                        tf.train.Feature(int64_list=tf.train.Int64List(value=self.roadgraph_id.flatten())),
                    "roadgraph/type":
                        tf.train.Feature(int64_list=tf.train.Int64List(value=self.roadgraph_type.flatten())),
                    "scenario_id":
                        tf.train.Feature(bytes_list=tf.train.BytesList(value=[self.scenario_id.encode("utf-8")])),
                }
                # fmt: on
            )
        )
        return tf_sample.SerializeToString()


def construct_spline(
    osm_data: ElementTree.ElementTree, way: ElementTree.Element
) -> InteractionSpline | None:
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


def parse_csv_to_datasets(csv_file: Path) -> List[InteractionRecord]:
    """Convert raw interaction data to list of records"""
    print(f"Started {csv_file.name}")

    # Transform csv data into scenarios
    tf_datasets: Dict[str, InteractionRecord] = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        while line := f.readline().strip():
            sample = InteractionSample.from_csv_line(line)
            scenario_id = csv_file.stem + str(sample.case)
            if scenario_id not in tf_datasets:
                tf_datasets[scenario_id] = InteractionRecord(scenario_id)
            tf_datasets[scenario_id].add_sample(sample)

    # Add roadgraph to scenarios
    imap = parse_map_osm(get_map_path(csv_file))
    for dataset in tf_datasets.values():
        dataset.add_roadgraph_tf(imap)

    print(f"Finished Parsing: {csv_file.name}")

    # Return list of scenarios
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
    for csv_file in split_folder.glob("*.csv"):
        tf_datasets.extend(parse_csv_to_datasets(csv_file))

    # Write Dataset to TF Record
    with tf.io.TFRecordWriter(str(output_rec)) as filew:
        with LivePbar(
            total=len(tf_datasets), desc=f"Serialising {split_folder.name} tfrecord"
        ) as pbar:
            for tf_dataset in tf_datasets:
                filew.write(tf_dataset.tf_serialise())
                pbar.update(1)


def run_parallel(data_root: Path, output_path: Path) -> None:
    """Run dataset generation for each split in parallel thread pool"""
    mp = get_context("spawn").Pool(processes=2)
    for split in ["train", "val"]:
        split_folder = data_root / split
        record_path = output_path / f"interaction_{split}.tfrecord"
        mp.apply_async(split_to_record, [split_folder, record_path])
    mp.close()
    mp.join()


def run_serial(data_root: Path, output_path: Path) -> None:
    """Run dataset generation in single thread"""
    for split in ["val", "train"]:
        split_folder = data_root / split
        record_path = output_path / f"interaction_{split}.tfrecord"
        split_to_record(split_folder, record_path)


@app.command()
def convert_dataset(
    src: Path,
    dst: Annotated[Optional[Path], typer.Argument()] = None,
    parallel: Annotated[bool, typer.Option()] = True,
) -> None:
    """Converts fragmented CSVs to single TFRecord Database for Training"""

    if dst is None:
        dst = src / "tfrecord"
        dst.mkdir(exist_ok=True)

    if parallel:
        run_parallel(src, dst)
    else:
        run_serial(src, dst)


if __name__ == "__main__":
    app()
