#!/usr/bin/env python
from enum import Enum
import json
import syn_infra as syn
import time
import os

def get_json_data(file_path):
    with open(file_path) as f:
        return json.load(f)


def get_node_tensors_names(data):
    return (
        set(data["input_tensors"]),
        set(data["output_tensors"]),
    )


class GlobalConfig():
    def __init__(self, args):
        self.json_files: List[str] = args.json_files
        self.models: str = args.models
        self.output: str = args.output
        self.graph_index: int = args.graph_index
        self.radius: int = args.radius
        self.single_node: bool = args.single_node
        self.nodes: List[str] = args.nodes
        self.guids: List[str] = args.guids
        self.bundles: List[str] = args.bundles
        self.all_bundles: bool = args.all_bundles
        self.post_graph: str = args.post_graph
        self.validate: bool = args.validate
        self.no_unique: bool = args.no_unique
        self.fwd: bool = args.fwd
        self.bwd: bool = args.bwd
        self.chip_type: str = args.chip_type
        self.venv: str = args.venv
        self.no_record: bool = args.no_record
        self.make_unique: bool = args.make_unique


class Config():
    def __init__(self, json_file: str, global_config: GlobalConfig):
        self.global_config: GlobalConfig = global_config
        self.json_file: str = json_file
        self.json_data = get_json_data(self.json_file)


class ScopedFile:
    def __init__(self, file_path: str, delete: bool) -> None:
        self.file_path = file_path
        self.delete = delete

    def __del__(self):
        if self.delete and os.path.exists(os.path.abspath(self.file_path)):
            os.remove(self.file_path)


class Direction(Enum):
    FORWARD = 1
    BACKWARD = 2

class Node:
    def __init__(self, node_name: str, all_nodes):
        self.consumers: set[str] = []
        self.producers: set[str] = []
        self.data: dict = all_nodes[node_name]
        self.name: str = node_name
        self.input_tensors: set[str] = None
        self.output_tensors: set[str] = None
        (
            self.input_tensors,
            self.output_tensors,
        ) = get_node_tensors_names(self.data)
        self._set_neighbors(all_nodes)

    def _set_neighbors(self, all_nodes):
        for n, d in all_nodes.items():
            if n == self.name:
                continue
            cit, cot = get_node_tensors_names(d)
            if self.input_tensors & cot:
                self.producers.append(n)
            if self.output_tensors & cit:
                self.consumers.append(n)

class GraphsLoader:
    def __init__(self, config: Config, log=None):
        self.config: Config = config
        self.log=log

        gc = config.global_config
        gc.origin_graph_index = gc.graph_index
        if (gc.no_record == False):
            self.pre_graphs_scoped_file = GraphsLoader.record_and_save_graphs("pre", config.json_file, gc.chip_type, gc.graph_index, gc.venv, self.log)
            self.post_graphs_scoped_file = GraphsLoader.record_and_save_graphs("post", self.pre_graphs_scoped_file.file_path, gc.chip_type, None, gc.venv, self.log)
            config.json_file = self.pre_graphs_scoped_file.file_path
            gc.post_graph = self.post_graphs_scoped_file.file_path
            if (gc.graph_index != None):
                gc.graph_index = 0
        self.pre_graphs = get_json_data(config.json_file).get("graphs")
        self.post_graphs = get_json_data(gc.post_graph).get("graphs")
        if (gc.graph_index != None):
            self.pre_graphs = [self.pre_graphs[int(gc.graph_index)]]
            self.post_graphs = [self.post_graphs[int(gc.graph_index)]]
            self.pre_graphs[0]["origin_graph_index"] = gc.origin_graph_index
            self.post_graphs[0]["origin_graph_index"] = gc.origin_graph_index


    @classmethod
    def record_and_save_graphs(cls, graphs_type, json_file, chip_type, graph_index, venv, log=None) -> ScopedFile:
        graphs_new_path = f'/tmp/ge_{graphs_type}_graphs-{time.strftime("%Y%m%d-%H%M%S")}'
        GraphsLoader.record_graph(json_file, graphs_new_path, graphs_type, chip_type, graph_index, venv, log)
        return ScopedFile(os.path.abspath(f"{graphs_new_path}.json"), True)

    @classmethod
    def record_graph(cls, graph_to_record_path, new_graph_path, graphs_type, chip_type, graph_index, venv=None, log=None):
        cmd = [
            "synrec",
            "-p",
            new_graph_path,
            "--graph-states",
            graphs_type,
            "--",
            "run_from_json",
            "-r",
            "-j",
            graph_to_record_path,
            "-c",
            chip_type,
        ]
        if graph_index is not None:
            cmd += ["-g", str(graph_index)]
        if (log != None):
            log.debug(f"executing cmd in external bash: {' '.join(cmd)}")
        syn.run_external_in_bash(cmd, venv)
        if not os.path.exists(os.path.abspath(f"{new_graph_path}.json")):
            raise RuntimeError("Failed to record post graph")
