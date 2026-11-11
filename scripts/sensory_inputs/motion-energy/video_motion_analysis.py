"""Extract optical-flow motion magnitude from consecutive video frames.

The sensory-control batch uses whole-frame Farneback flow and saves a headerless
CSV, including a zero for the first frame. Plotting is optional and does not
change the saved measurements. See ../README.md for the study settings.
"""

import cv2
import sys
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple


def extract_motion_timeseries(
    video_path: str,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    flow_method: str = 'farneback',
    show_progress: bool = True
) -> np.ndarray:
    """
    Extract a motion time series from an MP4 video.
    
    This function computes optical flow between consecutive frames and returns
    a 1D array representing the total motion magnitude at each time step.
    
    Parameters:
    -----------
    video_path : str
        Path to the input MP4 video file
    bbox : Optional[Tuple[int, int, int, int]]
        Optional bounding box (x, y, width, height) to restrict motion analysis
        to a specific region of the video. If None, analyzes the entire frame.
    flow_method : str
        Method for optical flow computation. Options:
        - 'farneback': Gunnar Farneback's algorithm (default, dense flow)
        - 'lucaskanade': Lucas-Kanade sparse optical flow
    show_progress : bool
        If True, print progress updates during processing (default: True)

    Returns:
    --------
    np.ndarray
        1D array of shape (T,) where T is the number of frames in the video.
        Each value represents the total motion magnitude at that frame.
        The first frame has motion value of 0 (no previous frame to compare).
    
    Example:
    --------
    >>> motion = extract_motion_timeseries('video.mp4')
    >>> print(f"Video has {len(motion)} frames")
    >>> print(f"Maximum motion at frame {np.argmax(motion)}")
    
    >>> # Analyze motion only in a specific region
    >>> motion_roi = extract_motion_timeseries('video.mp4', bbox=(100, 100, 300, 200))
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize motion time series
    motion_timeseries = np.zeros(total_frames)
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame from video")
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply bounding box if specified
    if bbox is not None:
        x, y, w, h = bbox
        prev_gray_roi = prev_gray[y:y+h, x:x+w]
    else:
        prev_gray_roi = prev_gray
    
    print("Extracting motion time series...")
    frame_idx = 1
    pbar = tqdm(total=total_frames - 1, desc="Processing", disable=(not show_progress) or (not sys.stderr.isatty()))

    # Process each subsequent frame
    while True:
        ret, curr_frame = cap.read()

        if not ret:
            break
        
        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply bounding box if specified
        if bbox is not None:
            x, y, w, h = bbox
            curr_gray_roi = curr_gray[y:y+h, x:x+w]
        else:
            curr_gray_roi = curr_gray
        
        # Compute optical flow
        if flow_method == 'farneback':
            # Farneback optical flow (dense)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray_roi,
                curr_gray_roi,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
            
            # Calculate magnitude of flow vectors
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Sum of all motion magnitudes in this frame
            total_motion = np.sum(magnitude)
            
        elif flow_method == 'lucaskanade':
            # Lucas-Kanade sparse optical flow
            # Detect features to track
            feature_params = dict(
                maxCorners=100,
                qualityLevel=0.3,
                minDistance=7,
                blockSize=7
            )
            
            p0 = cv2.goodFeaturesToTrack(prev_gray_roi, mask=None, **feature_params)
            
            if p0 is not None:
                # Calculate optical flow
                lk_params = dict(
                    winSize=(15, 15),
                    maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray_roi, curr_gray_roi, p0, None, **lk_params
                )
                
                # Select good points
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    # Calculate total displacement
                    displacements = np.linalg.norm(good_new - good_old, axis=1)
                    total_motion = np.sum(displacements)
                else:
                    total_motion = 0.0
            else:
                total_motion = 0.0
        else:
            raise ValueError(f"Unknown flow method: {flow_method}")
        
        motion_timeseries[frame_idx] = total_motion
        pbar.update(1)

        # Update previous frame
        prev_gray_roi = curr_gray_roi

        frame_idx += 1
    
    cap.release()
    pbar.close()

    # Verify we processed all frames
    if frame_idx != total_frames:
        # Trim to actual processed frames
        motion_timeseries = motion_timeseries[:frame_idx]
    
    return motion_timeseries


def plot_motion_timeseries(
    motion: np.ndarray,
    fps: float = 30.0,
    title: str = "Motion Analysis",
    save_path: Optional[str] = None
):
    """
    Plot the motion time series.
    
    Parameters:
    -----------
    motion : np.ndarray
        Motion time series from extract_motion_timeseries
    fps : float
        Frames per second of the video (for time axis)
    title : str
        Title for the plot
    save_path : Optional[str]
        If provided, save the plot to this path
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    motion_vals = motion.copy()
    motion_vals[motion_vals > 0.4 * 1e6] = np.nan  # Mask extreme outliers
    time_seconds = np.arange(len(motion_vals)) / fps

    fig = plt.figure(figsize=(9,3), dpi=300)
    plt.plot(time_seconds / 60, motion_vals, linewidth=1)
    plt.xlabel('Time (min.)')
    plt.ylabel('Motion Magnitude (a.u.)')
    plt.xticks(np.arange(0, time_seconds[-1]/60 + 1, 1), rotation=90)
    plt.yticks([10])

    plt.xlim([-0.5, time_seconds[-1]/60 + 0.5])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    import os.path

    parser = argparse.ArgumentParser(
        description="Extract motion time series from video using optical flow."
    )
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Bounding box to restrict analysis (x y width height)",
    )
    parser.add_argument(
        "--method",
        choices=["farneback", "lucaskanade"],
        default="farneback",
        help="Optical flow method (default: farneback)",
    )
    parser.add_argument(
        "--save-data",
        metavar="PATH",
        help="Save motion data to file (.npy or .csv)",
    )
    parser.add_argument(
        "--save-plot",
        metavar="PATH",
        help="Save plot to file instead of displaying",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS for plot time axis (default: read from video)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting entirely",
    )

    args = parser.parse_args()

    bbox = tuple(args.bbox) if args.bbox else None
    if bbox:
        print(f"Analyzing motion in bounding box: {bbox}")

    print(f"Analyzing video: {args.video_path}")
    if not args.save_data or not os.path.exists(args.save_data):
        motion = extract_motion_timeseries(
            args.video_path, bbox=bbox, flow_method=args.method
        )
    else:
        print(f"Loading existing motion data from {args.save_data}")
        if args.save_data.endswith(".csv"):
            motion = np.loadtxt(args.save_data, delimiter=",")
        else:
            motion = np.load(args.save_data)

    print(f"\nResults:")
    print(f"Total frames: {len(motion)}")
    print(f"Mean motion: {np.mean(motion):.2f}")
    print(f"Max motion: {np.max(motion):.2f} at frame {np.argmax(motion)}")
    print(f"Std motion: {np.std(motion):.2f}")

    # Save motion data if requested
    if args.save_data and not os.path.exists(args.save_data):
        if args.save_data.endswith(".csv"):
            np.savetxt(args.save_data, motion, delimiter=",")
        else:
            np.save(args.save_data, motion)
        print(f"Motion data saved to {args.save_data}")

    # Plot unless --no-plot
    if not args.no_plot:
        # Get FPS from video if not specified
        fps = args.fps
        if fps is None:
            cap = cv2.VideoCapture(args.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

        plot_motion_timeseries(
            motion,
            fps=fps,
            title=f"Motion Analysis: {args.video_path}",
            save_path=args.save_plot,
        )
