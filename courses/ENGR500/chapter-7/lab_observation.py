import json
from pathlib import Path

OUTPUT_PATH = Path("observations.json")


def make_observation(timestamp, image_name, visible_stars, imu_packet):
    return {
        "timestamp": timestamp,
        "image_name": image_name,
        "visible_stars": visible_stars,
        "imu_packet": imu_packet,
    }


def build_session():
    return [
        make_observation(
            "02:01:24",
            "IMG_0001.jpg",
            ["Polaris", "Vega", "Deneb"],
            {"time": 0.01, "roll": 0.11, "pitch": -0.01, "yaw": 182.0},
        ),
        make_observation(
            "02:01:29",
            "IMG_0002.jpg",
            ["Polaris", "Vega"],
            {"time": 5.02, "roll": 0.13, "pitch": -0.02, "yaw": 182.4},
        ),
        make_observation(
            "02:01:34",
            "IMG_0003.jpg",
            ["Polaris", "Deneb", "Altair"],
            {"time": 10.03, "roll": 0.10, "pitch": 0.00, "yaw": 183.1},
        ),
    ]


def save_session(observations, path=OUTPUT_PATH):
    with open(path, "w") as f:
        json.dump(observations, f, indent=2)


def load_session(path=OUTPUT_PATH):
    with open(path, "r") as f:
        return json.load(f)


def print_summary(observations):
    print(f"Session contains {len(observations)} observations\n")
    for i, obs in enumerate(observations):
        stars = ", ".join(obs["visible_stars"])
        imu = obs["imu_packet"]
        print(f"[{i}] {obs['timestamp']}  {obs['image_name']}")
        print(f"    stars: {stars}")
        print(
            f"    imu: t={imu['time']}s roll={imu['roll']} "
            f"pitch={imu['pitch']} yaw={imu['yaw']}"
        )


if __name__ == "__main__":
    session = build_session()
    save_session(session)
    print(f"Saved {len(session)} observations to {OUTPUT_PATH.resolve()}\n")

    reloaded = load_session()
    print_summary(reloaded)
