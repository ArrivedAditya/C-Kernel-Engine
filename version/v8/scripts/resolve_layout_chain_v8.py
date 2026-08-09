#!/usr/bin/env python3
"""Mechanical physical-layout compatibility and provider-chain ranking.

Circuits own logical edges. Kernel maps own physical layouts, placement
capabilities, and provider priority. This module joins those declarations; it
does not contain model or operation-family policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class LayoutEndpoint:
    provider_id: str
    port: str
    layout: str
    placement: str
    priority: int


@dataclass(frozen=True)
class LayoutRoute:
    producer: LayoutEndpoint
    consumer: LayoutEndpoint
    converter_id: Optional[str]
    conversion_cost: int

    @property
    def rank(self) -> tuple[int, int, str, str, str]:
        return (
            self.conversion_cost,
            -(self.producer.priority + self.consumer.priority),
            self.producer.provider_id,
            self.consumer.provider_id,
            self.converter_id or "",
        )


def _selection_priority(provider: Dict[str, Any]) -> int:
    selection = provider.get("selection")
    if not isinstance(selection, dict):
        return 0
    priority = selection.get("priority", 0)
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise RuntimeError(
            f"provider {provider.get('id')!r} has non-integer selection priority"
        )
    return priority


def _port_endpoint(
    provider: Dict[str, Any],
    *,
    field: str,
    port_name: str,
) -> LayoutEndpoint:
    ports = provider.get(field)
    if not isinstance(ports, list):
        raise RuntimeError(f"provider {provider.get('id')!r} has no {field} ports")
    matches = [port for port in ports if isinstance(port, dict) and port.get("name") == port_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"provider {provider.get('id')!r} must declare exactly one {field} port {port_name!r}"
        )
    port = matches[0]
    layout = str(port.get("layout", "") or "").strip()
    if not layout:
        raise RuntimeError(
            f"provider {provider.get('id')!r} port {port_name!r} has no physical layout"
        )
    placement = str(port.get("placement", "local") or "local").strip()
    return LayoutEndpoint(
        provider_id=str(provider.get("id", "")),
        port=port_name,
        layout=layout,
        placement=placement,
        priority=_selection_priority(provider),
    )


def _converter_for(
    converters: Sequence[Dict[str, Any]],
    source: LayoutEndpoint,
    destination: LayoutEndpoint,
) -> tuple[Optional[str], int]:
    if source.layout == destination.layout and source.placement == destination.placement:
        return None, 0
    matches = []
    for converter in converters:
        capability = converter.get("layout_conversion", converter)
        if not isinstance(capability, dict):
            continue
        if capability.get("from_layout") != source.layout:
            continue
        if capability.get("to_layout") != destination.layout:
            continue
        from_placement = capability.get("from_placement", source.placement)
        to_placement = capability.get("to_placement", destination.placement)
        if from_placement != source.placement or to_placement != destination.placement:
            continue
        cost = capability.get("cost_rank")
        if isinstance(cost, bool) or not isinstance(cost, int) or cost <= 0:
            raise RuntimeError(f"layout converter {converter.get('id')!r} requires positive cost_rank")
        matches.append((cost, str(converter.get("id", ""))))
    if not matches:
        raise RuntimeError(
            "no physical-layout route from "
            f"{source.provider_id}:{source.port}/{source.layout}/{source.placement} to "
            f"{destination.provider_id}:{destination.port}/{destination.layout}/{destination.placement}"
        )
    matches.sort()
    if len(matches) > 1 and matches[0][0] == matches[1][0]:
        raise RuntimeError(
            f"ambiguous equal-cost layout converters: {matches[0][1]}, {matches[1][1]}"
        )
    return matches[0][1], matches[0][0]


def rank_layout_routes(
    producers: Iterable[Dict[str, Any]],
    *,
    producer_port: str,
    consumers: Iterable[Dict[str, Any]],
    consumer_port: str,
    converters: Sequence[Dict[str, Any]] = (),
) -> List[LayoutRoute]:
    """Return compatible producer/consumer chains, cheapest and highest priority first."""
    routes: List[LayoutRoute] = []
    failures: List[str] = []
    for producer in producers:
        source = _port_endpoint(producer, field="outputs", port_name=producer_port)
        for consumer in consumers:
            destination = _port_endpoint(consumer, field="inputs", port_name=consumer_port)
            try:
                converter_id, cost = _converter_for(converters, source, destination)
            except RuntimeError as exc:
                failures.append(str(exc))
                continue
            routes.append(LayoutRoute(source, destination, converter_id, cost))
    routes.sort(key=lambda route: route.rank)
    if not routes:
        detail = "; ".join(failures) if failures else "no providers supplied"
        raise RuntimeError(f"no compatible physical provider chain: {detail}")
    if len(routes) > 1 and routes[0].rank[:2] == routes[1].rank[:2]:
        raise RuntimeError(
            "ambiguous equal-rank physical provider chains: "
            f"{routes[0].producer.provider_id}->{routes[0].consumer.provider_id}, "
            f"{routes[1].producer.provider_id}->{routes[1].consumer.provider_id}"
        )
    return routes
