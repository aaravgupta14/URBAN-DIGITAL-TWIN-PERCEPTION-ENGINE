import argparse
import json
import os
import subprocess
import sys

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
CAMERAS_DIR = os.path.join(ROOT, "cameras")
ANALYSES = ["sanity_checks_Generalized.py", "eda_Generalized.py", "ttc_Generalized.py", "eval_Generalized.py"]


def script(name):
    return os.path.join(ROOT, name)


def probe_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source: {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if width == 0 or height == 0:
        ok, frame = cap.read()
        if ok:
            height, width = frame.shape[:2]
    cap.release()
    return width, height, fps


def write_calibration(folder, video, fps, start, end):
    path = os.path.join(folder, "calibration.json")
    calib = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            calib = json.load(f)
    calib.update({"fps": fps, "video": video})
    if start is not None or end is not None:
        calib["frame_range"] = [start or 0, end]
    else:
        calib.pop("frame_range", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)


def step(title):
    print(f"\n========== {title} ==========")


def run(cmd, folder, capture=False):
    result = subprocess.run([sys.executable] + cmd, cwd=folder, capture_output=capture, text=True)
    if result.returncode != 0:
        if capture:
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])
        raise SystemExit(f"step failed: {os.path.basename(cmd[0])}")
    return result.stdout if capture else ""


def main():
    parser = argparse.ArgumentParser(description="Set up and run the digital twin for a calibrated camera")
    parser.add_argument("--name", required=True, help="camera name; files go to cameras/<name>")
    parser.add_argument("--video", required=True, help="video file or stream URL (rtsp://...)")
    parser.add_argument("--camera-height", type=float, help="lens height above the road in metres")
    parser.add_argument("--pitch", type=float, help="downward tilt of the camera in degrees")
    lens = parser.add_mutually_exclusive_group()
    lens.add_argument("--focal", type=float, help="focal length in pixels (fx of the calibration matrix)")
    lens.add_argument("--hfov", type=float, help="horizontal field of view in degrees")
    parser.add_argument("--fps", type=float, help="frame rate; read from the video if omitted")
    parser.add_argument("--speed-limit", type=float, help="overspeed limit in km/h")
    parser.add_argument("--start", type=int, help="first frame to process (files only)")
    parser.add_argument("--end", type=int, help="last frame to process (files only)")
    parser.add_argument("--fast", action="store_true", help="faster tracking without camera-motion compensation")
    parser.add_argument("--no-3d", action="store_true", help="turn off the live 3D view")
    parser.add_argument("--no-show", action="store_true", help="no live windows")
    parser.add_argument("--recalibrate", action="store_true", help="rebuild the homography even if it exists")
    parser.add_argument("--skip-tracking", action="store_true", help="only calibrate")
    parser.add_argument("--skip-analysis", action="store_true", help="do not run the offline reports")
    args = parser.parse_args()

    folder = os.path.join(CAMERAS_DIR, args.name)
    os.makedirs(folder, exist_ok=True)
    video = os.path.abspath(args.video) if os.path.exists(args.video) else args.video

    step("1/4 Camera and video")
    width, height, probed_fps = probe_video(video)
    fps = args.fps or probed_fps
    if not fps or fps <= 0:
        raise SystemExit("could not read the frame rate from the video; pass --fps")
    print(f"folder     : {folder}")
    print(f"video      : {video}")
    print(f"resolution : {width} x {height}   fps: {fps:.2f}{' (from video)' if not args.fps else ''}")
    write_calibration(folder, video, fps, args.start, args.end)

    step("2/4 Ground-plane calibration")
    have_homography = os.path.exists(os.path.join(folder, "homography.npy"))
    if have_homography and not args.recalibrate:
        print("homography.npy already exists; reusing it (pass --recalibrate to rebuild)")
    else:
        if args.camera_height is None or args.pitch is None or (args.focal is None and args.hfov is None):
            raise SystemExit("calibration needs --camera-height, --pitch and --focal or --hfov")
        lens_arg = ["--focal", str(args.focal)] if args.focal is not None else ["--hfov", str(args.hfov)]
        run([script("ground_model_Generalized.py"), "--camera-height", str(args.camera_height),
             "--pitch", str(args.pitch), *lens_arg, "--image-size", str(width), str(height), "--write"], folder)

    if args.skip_tracking:
        print("\nCalibration done; tracking skipped.")
        return

    step("3/4 Live digital twin with alerts")
    cmd = [script("digital_twin_log_Generalized.py")]
    if args.speed_limit is not None:
        cmd += ["--speed-limit", str(args.speed_limit)]
    if args.fast:
        cmd.append("--fast")
    if args.no_3d:
        cmd += ["--plot3d-every", "0"]
    if args.no_show:
        cmd.append("--no-show")
    run(cmd, folder)

    if not args.skip_analysis:
        step("4/4 Offline analysis")
        report = ""
        for name in ANALYSES:
            report += f"\n##### {name}\n" + run([script(name)], folder, capture=True)
        report += "\n##### database summary\n" + run([script("twin_db_Generalized.py"), "--summary"], folder, capture=True)
        with open(os.path.join(folder, "analysis.txt"), "w", encoding="utf-8") as f:
            f.write(report)
        print(report[-2500:])

    print(f"\nDone. Results in {folder}:")
    for name in ("alerts.csv", "twin_state.db", "tracking_log.csv", "analysis.txt", "output_tracking_source.avi",
                 "output_digital_twin.avi", "output_digital_twin_3d.avi"):
        if os.path.exists(os.path.join(folder, name)):
            print(f"  {name}")


if __name__ == "__main__":
    main()
