import sys
import threading
import uuid
from pathlib import Path

# Add the repo root to sys.path so we can import the core railmind engine
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from railmind.data_loader import load_railway
from railmind.graph import build_network_from_data
from railmind.twin import DigitalTwin

# In-memory store for twin sessions. The default twin is built lazily on the
# first request (never at import time) so the server always boots instantly,
# even with no network access.
twin_sessions = {}
_twin_lock = threading.Lock()


def _build_default_twin():
    stations, tracks = load_railway("India")
    graph = build_network_from_data(stations, tracks)
    twin = DigitalTwin(graph)
    twin.seed_trains(5)
    return twin


def get_twin(request):
    """
    Fetch the twin for this session, creating the default twin on first use.

    Args:
        request: The HTTP request object containing headers.

    Returns:
        DigitalTwin: The digital twin session instance if found, otherwise None.
    """
    session_id = request.headers.get("X-Session-ID", "default")
    with _twin_lock:
        if session_id == "default" and "default" not in twin_sessions:
            twin_sessions["default"] = _build_default_twin()
        return twin_sessions.get(session_id)


@api_view(['GET'])
def get_state(request):
    """
    Retrieve the current state of the Digital Twin network.

    This includes weather conditions, track statuses, train locations, and the graph structure.
    Used by all agents.

    Example usage:
    ```python
    get_state()
    ```

    Returns:
    ```json
    {
        "weather": {},
        "tracks": {},
        "trains": {},
        "graph": {}
    }
    ```
    """
    twin = get_twin(request)
    if not twin:
        return Response({"error": "Invalid session"}, status=status.HTTP_404_NOT_FOUND)

    state_dump = twin.get_state()

    graph_state = {
        "nodes": list(twin.graph.graph.nodes()),
        "edges": list(twin.graph.graph.edges())
    }

    return Response({
        "weather": state_dump.get("weather", {}),
        "tracks": state_dump.get("tracks", {}),
        "trains": state_dump.get("trains", {}),
        "stations": state_dump.get("stations", {}),
        "graph": graph_state
    })


@api_view(['POST'])
def reset_twin(request):
    """
    Rebuild the default twin from the canonical network data.

    Clears all sandbox sessions. Handy between demo runs.
    """
    with _twin_lock:
        twin_sessions.clear()
        twin_sessions["default"] = _build_default_twin()
    return Response({"status": "success", "message": "Digital twin reset to baseline state."})


@api_view(['POST'])
def copy_twin(request):
    """
    Create a new isolated sandbox session (a parallel future) based on the current state.

    Required for simulating multiple futures.

    Example usage:
    ```python
    future = twin.copy()
    ```
    """
    twin = get_twin(request)
    if not twin:
        return Response({"error": "Invalid session"}, status=status.HTTP_404_NOT_FOUND)

    new_twin = twin.copy()
    new_session_id = str(uuid.uuid4())
    with _twin_lock:
        twin_sessions[new_session_id] = new_twin

    return Response({
        "status": "success",
        "session_id": new_session_id,
        "message": "Future state created successfully."
    })


@api_view(['POST'])
def close_track(request):
    """
    Close a specific railway track due to maintenance or emergency.

    Used by Track Agent.

    Example usage:
    ```python
    close_track("T14")
    ```
    """
    twin = get_twin(request)
    if not twin:
        return Response({"error": "Invalid session"}, status=status.HTTP_404_NOT_FOUND)

    track_id = request.data.get("track_id")
    if not track_id:
        return Response({"error": "Missing track_id"}, status=status.HTTP_400_BAD_REQUEST)

    if track_id not in twin.state.tracks:
        return Response({"error": f"Track '{track_id}' not found"}, status=status.HTTP_404_NOT_FOUND)

    try:
        # Utilizing existing string-based apply_action for closure
        twin.apply_action(f"close_track_{track_id}")
        return Response({"status": "success", "track_id": track_id})
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def find_route(request):
    """
    Calculate the optimal route between a source and a destination station.

    Returns alternative route while avoiding closed tracks.
    Used by Routing Agent.

    Example usage:
    ```python
    find_route("A", "C")
    ```
    """
    twin = get_twin(request)
    if not twin:
        return Response({"error": "Invalid session"}, status=status.HTTP_404_NOT_FOUND)

    source = request.data.get("source")
    destination = request.data.get("destination")

    if not source or not destination:
        return Response({"error": "Missing source or destination"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        route = twin.graph.find_route(source, destination)
    except ValueError as e:
        return Response({"route": None, "error": str(e)}, status=status.HTTP_404_NOT_FOUND)

    return Response({"route": route, "status": "success"})


@api_view(['POST'])
def reroute_train(request):
    """
    Manually assign a new route to an existing train.

    Assigns a new route.
    Used by Planner/Simulation.

    Example usage:
    ```python
    reroute_train("TR01", ["A", "B", "C"])
    ```
    """
    twin = get_twin(request)
    if not twin:
        return Response({"error": "Invalid session"}, status=status.HTTP_404_NOT_FOUND)

    train_id = request.data.get("train_id")
    route = request.data.get("route")

    if not train_id or not route or not isinstance(route, list):
        return Response({"error": "Missing train_id or invalid route array"}, status=status.HTTP_400_BAD_REQUEST)

    if train_id not in twin.state.trains:
        return Response({"error": "Train ID not found"}, status=status.HTTP_404_NOT_FOUND)

    twin.reroute_train(train_id, route)
    return Response({"status": "success", "train_id": train_id, "route": route})


@api_view(['POST'])
def apply_action(request):
    """
    Apply a generic string-based action command to the twin state.

    Converts plan actions into state changes.

    Example usage:
    ```python
    apply_action("close_track_T14")

    apply_action("reroute_via_route_A")
    ```
    """
    twin = get_twin(request)
    if not twin:
        return Response({"error": "Invalid session"}, status=status.HTTP_404_NOT_FOUND)

    action = request.data.get("action")
    if not action:
        return Response({"error": "Missing action string"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        if action.startswith("reroute_"):
            # Mock fallback for string-based rerouting
            return Response({"status": "acknowledged", "action": action, "note": "Use reroute_train endpoint for concrete routing."})

        twin.apply_action(action)
        return Response({"status": "success", "action": action})
    except ValueError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def calculate_delay(request):
    """
    Calculate the current total delay of the network based on train speeds and track closures.

    Returns:
    ```python
    delay_minutes
    ```
    """
    twin = get_twin(request)
    if not twin:
        return Response({"error": "Invalid session"}, status=status.HTTP_404_NOT_FOUND)

    delay = twin.calculate_delay()
    return Response({"delay_minutes": delay})


@api_view(['GET'])
def calculate_risk(request):
    """
    Calculate the current operational risk score of the network.

    Factors such as track health and bad weather contribute to the risk score.

    Returns:
    ```python
    risk_score
    ```
    """
    twin = get_twin(request)
    if not twin:
        return Response({"error": "Invalid session"}, status=status.HTTP_404_NOT_FOUND)

    risk = twin.calculate_risk()
    return Response({"risk_score": risk})
