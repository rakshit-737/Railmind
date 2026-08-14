import time
import copy
import random
from .models import NetworkState, StationNode, TrackSegment, TrainState, TrackStatus, WeatherCondition
from .graph import RailwayGraph

class DigitalTwin:
    # One simulated minute counts as this many minutes of train movement,
    # so real inter-city distances (avg ~410 km) still cross in demo time.
    SIM_TIME_COMPRESSION = 45.0

    def __init__(self, graph: RailwayGraph):
        self.graph = graph
        stations = {}
        for n, data in graph.graph.nodes(data=True):
            stations[n] = StationNode(
                station_id=n,
                name=data.get("name", n),
                congestion_level=0.2,
                active_signals={}
            )
            
        tracks = {}
        for u, v, data in graph.graph.edges(data=True):
            t_id = data["track_id"]
            if t_id not in tracks:
                tracks[t_id] = TrackSegment(
                    track_id=t_id,
                    source=u,
                    destination=v,
                    health=data["health"],
                    status=data["status"],
                    length_km=data["length_km"],
                    max_speed_kmh=data["max_speed_kmh"]
                )
                
        self.state = NetworkState(
            weather={},
            tracks=tracks,
            trains={},
            stations=stations,
            timestamp=time.time()
        )
        self._rng = random.Random(2024)
        self._last_tick = time.time()
        # Manually injected weather is pinned: ambient decay must not clear it
        self._pinned_weather = set()

    def get_state(self):
        dump = self.state.model_dump()
        
        for st in dump["stations"].values():
            st["active_signals"] = {k: v.value if hasattr(v, "value") else v for k, v in st["active_signals"].items()}
        for tr in dump["tracks"].values():
            if hasattr(tr["status"], "value"):
                tr["status"] = tr["status"].value
        for k, w in dump["weather"].items():
            if hasattr(w, "value"):
                dump["weather"][k] = w.value
                
        return dump

    def copy(self):
        return copy.deepcopy(self)

    def close_track(self, track_id):
        if track_id in self.state.tracks:
            self.state.tracks[track_id].status = TrackStatus.CLOSED
            self.state.tracks[track_id].health = 0.0
        self.graph.close_track(track_id)

    def find_route(self, source, destination):
        return self.graph.find_route(source, destination)

    def reroute_train(self, train_id, route):
        if len(route) < 2:
            raise ValueError("A route needs at least 2 stations")
        if train_id in self.state.trains:
            train = self.state.trains[train_id]
            train.route = route
            train.current_station = route[0]
            train.route_index = 0
            train.progress = 0.0
            train.held = False

    def apply_action(self, action_string):
        """Apply a generic string-encoded action to the twin.

        Supported action grammar:
          close_track_<TRACK_ID>
              e.g. "close_track_T23"
          reroute_<TRAIN_ID>_via_<STATION>_<STATION>[_<STATION>...]
              e.g. "reroute_TR00_via_NEW_DELHI_JAIPUR_JUNCT"
              The part after "_via_" is two or more station ids joined by
              underscores. Station ids may themselves contain underscores;
              they are matched against the network with backtracking.

        Raises ValueError for unrecognised actions, unknown trains or
        stations, or routes with fewer than two stations.
        """
        if action_string.startswith("close_track_"):
            parts = action_string.split("close_track_")
            if len(parts) == 2:
                self.close_track(parts[1])
            else:
                raise ValueError(f"Unrecognised action string: {action_string}")
        elif action_string.startswith("reroute_"):
            train_id, route = self._parse_reroute_action(action_string)
            self.reroute_train(train_id, route)
        else:
            raise ValueError(f"Unrecognised action string: {action_string}")

    def _parse_reroute_action(self, action_string):
        """Parse "reroute_<TRAIN_ID>_via_<STATIONS>" into (train_id, route)."""
        body = action_string[len("reroute_"):]
        train_id, sep, route_part = body.partition("_via_")
        if not sep or not train_id or not route_part:
            raise ValueError(
                f"Malformed reroute action: {action_string!r} "
                "(expected reroute_<TRAIN_ID>_via_<STATION>_<STATION>...)"
            )
        if train_id not in self.state.trains:
            raise ValueError(f"Unknown train in reroute action: {train_id}")

        stations = set(self.graph.graph.nodes())
        tokens = route_part.split("_")

        def parse(i):
            if i == len(tokens):
                return []
            # Longest match first, backtracking on failure
            for j in range(len(tokens), i, -1):
                candidate = "_".join(tokens[i:j])
                if candidate in stations:
                    rest = parse(j)
                    if rest is not None:
                        return [candidate] + rest
            return None

        route = parse(0)
        if route is None:
            raise ValueError(
                f"Unknown station in reroute action: {route_part!r}"
            )
        if len(route) < 2:
            raise ValueError(
                f"Reroute action needs at least 2 stations, got: {route}"
            )
        return train_id, route

    def seed_trains(self, n):
        # Local RNG: keeps fleets deterministic without touching the
        # process-global random module.
        rng = random.Random(42)
        nodes = list(self.graph.graph.nodes())
        if len(nodes) < 2:
            return

        count = 0
        i = 0
        max_attempts = n * 50
        while count < n and i < max_attempts:
            src = rng.choice(nodes)
            dst = rng.choice(nodes)
            if src != dst:
                try:
                    route = self.find_route(src, dst)
                    if route:
                        t_id = f"TR{count:02d}"
                        self.state.trains[t_id] = TrainState(
                            train_id=t_id,
                            current_station=route[0],
                            route=route,
                            speed_kmh=80.0,
                            passengers=200 + (i * 37 % 300),
                            delayed_minutes=0.0
                        )
                        count += 1
                except ValueError:
                    pass
            i += 1

    def _track_between(self, u, v):
        for tr in self.state.tracks.values():
            if {tr.source, tr.destination} == {u, v}:
                return tr
        return None

    def set_weather(self, track_id, condition, pin: bool = True):
        """Set the weather condition on a track.

        Pinned conditions (the default, for operator injections) survive
        ambient decay in tick(). Ambient/scenario callers should pass
        pin=False so their conditions decay naturally and never accumulate.
        """
        if track_id not in self.state.tracks:
            raise ValueError(f"Unknown track: {track_id}")
        condition = WeatherCondition(condition)
        if condition == WeatherCondition.CLEAR:
            self.state.weather.pop(track_id, None)
            self._pinned_weather.discard(track_id)
        else:
            self.state.weather[track_id] = condition
            if pin:
                self._pinned_weather.add(track_id)

    def tick(self, minutes: float = 2.0):
        """Advance the living twin one step: trains move along their routes,
        congestion drifts, weather comes and goes. Closed tracks hold trains."""
        rng = self._rng

        for train in self.state.trains.values():
            route = train.route
            if len(route) < 2:
                continue

            # Reached the terminus: turn around and run the return service
            if train.route_index >= len(route) - 1:
                train.route = list(reversed(route))
                train.route_index = 0
                train.progress = 0.0
                train.current_station = train.route[0]
                continue

            next_station = route[train.route_index + 1]
            track = self._track_between(train.current_station, next_station)
            if track is None or track.status == TrackStatus.CLOSED:
                train.held = True
                train.delayed_minutes += minutes
                continue

            train.held = False
            # Physics-based step: distance covered this tick over track length.
            # SIM_TIME_COMPRESSION squeezes real inter-city travel times into
            # demo pace (avg step per 2-min tick stays near the old 0.30).
            speed_kmh = min(train.speed_kmh, track.max_speed_kmh)
            if track.status == TrackStatus.DEGRADED:
                speed_kmh *= 0.6
            step = (speed_kmh * (minutes * self.SIM_TIME_COMPRESSION) / 60.0) / max(track.length_km, 1.0)
            step = min(1.0, step * (0.8 + 0.4 * rng.random()))
            train.progress += step
            if track.status == TrackStatus.DEGRADED:
                train.delayed_minutes += minutes * 0.5

            if train.progress >= 1.0:
                train.route_index += 1
                train.current_station = next_station
                train.progress = 0.0

        for st in self.state.stations.values():
            st.congestion_level = min(0.95, max(0.05, st.congestion_level + rng.uniform(-0.05, 0.05)))

        track_ids = list(self.state.tracks.keys())
        if track_ids and rng.random() < 0.12:
            tid = rng.choice(track_ids)
            # Never overwrite an operator-injected condition
            if tid not in self._pinned_weather:
                self.state.weather[tid] = rng.choices(
                    [WeatherCondition.RAIN, WeatherCondition.FOG, WeatherCondition.STORM],
                    weights=[0.6, 0.25, 0.15],
                )[0]
        for tid in list(self.state.weather.keys()):
            if tid not in self._pinned_weather and rng.random() < 0.25:
                del self.state.weather[tid]

        self.state.timestamp = time.time()

    def maybe_tick(self, min_interval_s: float = 2.0, max_ticks: int = 5):
        """Lazy tick: advance the twin if enough wall-clock time has passed.
        Lets state reads drive the simulation without a background thread."""
        elapsed = time.time() - self._last_tick
        if elapsed < min_interval_s:
            return False
        for _ in range(min(int(elapsed / min_interval_s), max_ticks)):
            self.tick()
        self._last_tick = time.time()
        return True

    def calculate_delay(self):
        total = 0.0
        for tr in self.state.trains.values():
            total += tr.delayed_minutes
            for st_id in tr.route:
                if st_id in self.state.stations:
                    total += 5.0 * self.state.stations[st_id].congestion_level
        return total

    def calculate_risk(self):
        total = 0.0
        for tr in self.state.tracks.values():
            w = 0.5
            if tr.status == TrackStatus.CLOSED:
                w = 2.0
            elif tr.status == TrackStatus.DEGRADED:
                w = 1.5
            risk = (1.0 - tr.health) * w
            
            weather_val = self.state.weather.get(tr.track_id)
            if weather_val == WeatherCondition.STORM:
                risk += 0.3
            elif weather_val == WeatherCondition.RAIN:
                risk += 0.1
            elif weather_val == WeatherCondition.FOG:
                risk += 0.15
                
            total += risk
        return total
