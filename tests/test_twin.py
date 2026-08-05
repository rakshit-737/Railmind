import pytest

from railmind.data_loader import load_bundled_network
from railmind.graph import build_network_from_data
from railmind.models import TrackStatus
from railmind.twin import DigitalTwin


@pytest.fixture()
def twin():
    stations, tracks = load_bundled_network()
    graph = build_network_from_data(stations, tracks)
    t = DigitalTwin(graph)
    t.seed_trains(5)
    return t


def test_seed_trains_deterministic(twin):
    stations, tracks = load_bundled_network()
    other = DigitalTwin(build_network_from_data(stations, tracks))
    other.seed_trains(5)
    assert len(twin.state.trains) == 5
    for tid, train in twin.state.trains.items():
        assert other.state.trains[tid].route == train.route


def test_close_track_updates_state_and_graph(twin):
    risk_before = twin.calculate_risk()
    twin.close_track("T23")
    assert twin.state.tracks["T23"].status == TrackStatus.CLOSED
    assert twin.state.tracks["T23"].health == 0.0
    assert twin.calculate_risk() > risk_before


def test_find_route_avoids_closed_track(twin):
    # T23 is the only direct Nagpur-Raipur link; closing it forces a detour
    direct = twin.find_route("NAGPUR_JUNCT", "RAIPUR_JUNCT")
    assert direct == ["NAGPUR_JUNCT", "RAIPUR_JUNCT"]
    twin.close_track("T23")
    detour = twin.find_route("NAGPUR_JUNCT", "RAIPUR_JUNCT")
    assert len(detour) > 2
    for i in range(len(detour) - 1):
        assert {detour[i], detour[i + 1]} != {"NAGPUR_JUNCT", "RAIPUR_JUNCT"}


def test_tick_moves_trains(twin):
    positions_before = {
        tid: (t.route_index, t.progress) for tid, t in twin.state.trains.items()
    }
    for _ in range(3):
        twin.tick()
    moved = any(
        (t.route_index, t.progress) != positions_before[tid]
        for tid, t in twin.state.trains.items()
    )
    assert moved


def test_tick_holds_train_at_closed_track(twin):
    train = next(iter(twin.state.trains.values()))
    # Close the edge directly ahead of the train
    nxt = train.route[train.route_index + 1]
    track = twin._track_between(train.current_station, nxt)
    assert track is not None
    twin.close_track(track.track_id)

    delayed_before = train.delayed_minutes
    twin.tick()
    assert train.held is True
    assert train.delayed_minutes > delayed_before
    assert train.current_station == train.route[train.route_index]


def test_reroute_train_resets_position(twin):
    tid = next(iter(twin.state.trains.keys()))
    new_route = ["NEW_DELHI", "JAIPUR_JUNCT", "BHOPAL_JUNCT"]
    twin.reroute_train(tid, new_route)
    train = twin.state.trains[tid]
    assert train.route == new_route
    assert train.current_station == "NEW_DELHI"
    assert train.route_index == 0 and train.progress == 0.0


def test_reroute_train_rejects_degenerate_route(twin):
    tid = next(iter(twin.state.trains.keys()))
    with pytest.raises(ValueError):
        twin.reroute_train(tid, ["NEW_DELHI"])
    with pytest.raises(ValueError):
        twin.reroute_train(tid, [])


def test_find_route_unknown_station_raises_value_error(twin):
    with pytest.raises(ValueError):
        twin.find_route("DELHI", "MUMBAI")


def test_set_weather_validates_track(twin):
    twin.set_weather("T05", "STORM")
    assert twin.state.weather["T05"].value == "STORM"
    twin.set_weather("T05", "CLEAR")
    assert "T05" not in twin.state.weather
    with pytest.raises(ValueError):
        twin.set_weather("NOPE", "STORM")


def test_maybe_tick_respects_interval(twin):
    assert twin.maybe_tick(min_interval_s=9999) is False
    twin._last_tick = 0.0
    assert twin.maybe_tick(min_interval_s=1) is True
