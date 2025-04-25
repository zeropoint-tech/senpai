#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

import argparse
from dataclasses import dataclass
import logging
import re
import time
import os
from typing import List
import signal

NUMA_STAT_NODE_REGEX = re.compile(r"N(\d+)=(\d+)")

interrupted = False

# Define logger globally for availability in doctests
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()


def interrupt_handler(sig, frame):
    global interrupted
    interrupted = True


def h(x):
    """Translates a number of bytes to a human-readable string."""
    order = 0
    suffix = ["", "k", "M", "G", "T"]
    max_order = len(suffix) - 1
    while abs(x) > 1024 and order < max_order:
        x /= 1024.0
        order += 1
    return f"{x:.2f}{suffix[order]}"


@dataclass
class NumaStatEntry:
    name: str
    nodes: List[int]

    @classmethod
    def from_line(cls, line: str):
        """Create a stat entry from a line in memory.numa_stat of the cgroup

        >>> NumaStatEntry.from_line("anon N0=123 N1=567")
        NumaStatEntry(name='anon', nodes=[123, 567])
        >>> NumaStatEntry.from_line("file N0=123")
        NumaStatEntry(name='file', nodes=[123])
        >>> NumaStatEntry.from_line("file N0=123 N1=0 N2=1 N3=567")
        NumaStatEntry(name='file', nodes=[123, 0, 1, 567])
        """
        line = line.strip()
        name = line.split(" ")[0]
        node_matches = NUMA_STAT_NODE_REGEX.findall(line)
        assert len(node_matches) > 0, f"Failed to find node matches in line: {line}"
        nodes = []
        for idx, match in enumerate(node_matches):
            assert len(match) == 2, f"invalid match length: {match}"
            nid, value = int(match[0]), int(match[1])
            assert idx == nid, f"Non-consistent node indexes in line {line}"
            nodes.append(value)
        return cls(name=name, nodes=nodes)


class Cgroup(object):
    def __init__(self, path: str, ram_min_size: int, ram_nodes: List[int]):
        self.path = path
        self.ram_min_size = ram_min_size
        self.ram_nodes = ram_nodes
        self.numa_memory()  # Ensure NUMA stats are available
        logger.info("Setting memory.swap.max to 0")
        self.write("memory.swap.max", str(0))

    def ram_nodes_str(self) -> str:
        return "N" + ",".join(str(x) for x in self.ram_nodes)

    def mem_used_ram(self, memory: List[int]) -> int:
        total = 0
        for node in self.ram_nodes:
            total += memory[node]
        return total

    def mem_used_total(self, memory: List[int]) -> int:
        return sum(memory)

    def mem_used_other(self, memory: List[int]) -> int:
        return self.mem_used_total(memory) - self.mem_used_ram(memory)

    def do_reclaim(self, factor):
        memory = self.numa_memory()
        ram_used = self.mem_used_ram(memory)
        requested_reclaim_bytes = round(abs(ram_used * factor))
        logger.debug(f"request={h(requested_reclaim_bytes)}")

        assert factor <= 0, f"Cannot serve memory increase requests"

        if ram_used < self.ram_min_size:
            logger.debug(
                f"Skipping because {self.ram_nodes_str()}={h(ram_used)} < min={h(self.ram_min_size)}"
            )
            return

        reclaim_bytes = min(requested_reclaim_bytes, ram_used - self.ram_min_size)
        reclaim_bytes &= ~4095  # Page mask
        assert reclaim_bytes >= 0

        new_memory = int(ram_used + ram_used * factor)
        logger.info(
            f"reclaim={h(reclaim_bytes)} | "
            f"{self.ram_nodes_str()}(expected)={h(new_memory)} change={100*factor:.3f}%"
        )

        try:
            self.write("memory.reclaim", str(reclaim_bytes))
        except Exception as e:
            logger.warning(f"Reclaim failed: {e}")

    def numa_memory(self, kinds={"anon", "file"}) -> List[int]:
        """Return a list of the memory per node for the given page kinds from memory.numa_stat"""
        numa_stat = self.readlines("memory.numa_stat")
        memory_usage = []
        found_pages = kinds.copy()
        for line in numa_stat:
            if len(found_pages) == 0:
                break
            entry = NumaStatEntry.from_line(line)
            if entry.name in found_pages:
                found_pages.remove(entry.name)
                if not memory_usage:
                    memory_usage = entry.nodes.copy()
                else:
                    assert len(entry.nodes) == len(
                        memory_usage
                    ), f"Unexpected number of numa nodes in entry: {entry}"
                    for i in range(len(memory_usage)):
                        memory_usage[i] += entry.nodes[i]
        return memory_usage

    def memory_high(self):
        x = self.read("memory.high")
        if x == "max\n":
            return (1 << 64) - 1
        return int(x)

    def read(self, filename):
        with open(os.path.join(self.path, filename)) as f:
            return f.read()

    def readlines(self, filename):
        with open(os.path.join(self.path, filename)) as f:
            return f.readlines()

    def write(self, filename, value):
        with open(os.path.join(self.path, filename), "w") as f:
            f.write(value)


def adjustment_ratio(actual: float, target: float, coeff: float, limit: float):
    """Back off exponentially as we deviate from the target pressure.

    The coefficient defines how sensitive
    we are to fluctuations around the target value: when
    the coefficient is 10, the curve reaches the adjustment
    limit when pressure is ten times the target.

    >>> adjustment_ratio(20, 1, 20, 1)
    1.0
    >>> adjustment_ratio(20, 1, 40, 1)
    0.25
    >>> adjustment_ratio(20, 1, 40, 0.5)
    0.125
    >>> adjustment_ratio(1, 20, 40, 0.5)
    -0.125
    """
    assert actual > target
    err = actual / target
    adj = (err / coeff) ** 2
    adj = min(adj * limit, limit)
    adj = -adj
    logger.debug(f"adj={adj:.3f} | actual={actual:4.2f} target={target:4.2f}")
    return adj


class Senpai(object):
    def __init__(self, conf):
        self.conf = conf
        logger.info("Configuration:")
        for key, val in vars(conf).items():
            logger.info(f"  {key} = {val}")

        self.cgroup = Cgroup(self.conf.cgpath, self.conf.ram_min_size, conf.ram_nodes)

    def run(self):
        signal.signal(signal.SIGINT, interrupt_handler)
        while not interrupted:
            time.sleep(conf.interval)
            self.tick()

    def tick(self):
        memory = self.cgroup.numa_memory()
        total, ram, other = (
            self.cgroup.mem_used_total(memory),
            self.cgroup.mem_used_ram(memory),
            self.cgroup.mem_used_other(memory),
        )
        ram_ratio = 100 * ram / total if total > 0 else float("nan")

        logger.info(
            f"ram={ram_ratio:4.2f}% target={self.conf.ram_pct:4.2f}% | "
            f"{self.cgroup.ram_nodes_str()}={h(ram):<7} "
            f"N(other)={h(other):<7} total={h(total):<7} "
            f"min({self.cgroup.ram_nodes_str()})={h(self.conf.ram_min_size):<7}"
        )
        if ram_ratio > self.conf.ram_pct:
            adj = adjustment_ratio(
                ram_ratio,
                self.conf.ram_pct,
                self.conf.decrease_coeff,
                self.conf.decrease_max,
            )
            self.cgroup.do_reclaim(adj)


def float_mb_to_byte(mb: str) -> int:
    """Convert a floating-point MiB value to bytes

    >>> float_mb_to_byte(1)
    1048576
    >>> float_mb_to_byte(0.1)
    104857
    >>> float_mb_to_byte(0.01)
    10485
    >>> float_mb_to_byte(0.001)
    1048
    >>> float_mb_to_byte(10)
    10485760

    """
    return int(float(mb) * 2**20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    Use cgroup's memory.reclaim to keep the requested ratio of memory demoted to the slower memory tier.
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("cgpath", type=str)
    parser.add_argument(
        "--ram-min-size",
        "-m",
        type=float_mb_to_byte,
        default=100,
        help="min memory (MBs) to keep in RAM",
    )
    parser.add_argument(
        "--interval", type=int, default=1, help="adjustment interval (seconds)"
    )
    parser.add_argument(
        "--ram-pct",
        "-r",
        type=float,
        default=30,
        help="target percentage of memory to keep in RAM",
    )
    parser.add_argument(
        "--decrease-max",
        type=lambda x: float(x) / 100,
        default=0.01,
        help="maximum percentage of cgroup size reclaim in [0, 100]",
    )
    parser.add_argument(
        "--decrease-coeff",
        type=float,
        default=2,
        help="decrease coefficient >= 1; higher values make the steps smoother",
    )
    parser.add_argument(
        "--ram-nodes",
        type=int,
        nargs="+",
        default=[0],
        help="NUMA nodes of the RAM tier",
    )
    parser.add_argument("--level", "-l", type=str, default="INFO", help="logger level")

    conf = parser.parse_args()

    assert conf.ram_min_size > 0, "ram_min_size"
    assert conf.interval >= 1, "interval"
    assert conf.ram_pct > 0 and conf.ram_pct <= 100, "ram_pct"
    assert conf.decrease_max >= 0 and conf.decrease_max <= 100, "decrease_max"
    assert conf.decrease_coeff >= 0, "decrease_coeff"
    assert conf.ram_nodes, "ram_nodes"

    logger.setLevel(conf.level.upper())

    senpai = Senpai(conf)
    senpai.run()
