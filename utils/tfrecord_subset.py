"""Helper script to take a waymo-open-motion record and place into its own file"""
from pathlib import Path

import tensorflow as tf
import typer
from typing_extensions import Annotated

from waymo_open_dataset.utils.occupancy_flow_data import parse_tf_example

app = typer.Typer()


@app.command()
def main(
    input_folder: Annotated[Path, typer.Option()],
    output_file: Annotated[Path, typer.Option()],
    count: Annotated[int, typer.Option()] = 1,
):
    filenames = list(input_folder.glob("*.tfrecord*"))
    dataset = tf.data.TFRecordDataset(filenames).map(parse_tf_example).take(count)

    def serialize_example(example):
        features = {}
        for key, value in example.items():
            if tf.constant(value).dtype == tf.float32:
                features[key] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=value.numpy().flatten())
                )
            elif tf.constant(value).dtype == tf.int64:
                features[key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=value.numpy().flatten())
                )
            elif tf.constant(value).dtype == tf.string:
                features[key] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[value.numpy()])
                )
            else:
                raise NotImplementedError(
                    f"Unrecognised type: {tf.constant(value).dtype}"
                )
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()

    with tf.io.TFRecordWriter(str(output_file.resolve())) as writer:
        for sample in dataset:
            writer.write(serialize_example(sample))

    print("Done")


if __name__ == "__main__":
    app()
