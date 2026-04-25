# #!/usr/bin/python
# # -*- coding: utf-8 -*-
# """
# On-line Dynamic Time Warping
# """

# import time
# from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

# import numpy as np
# import progressbar
# from numpy.typing import NDArray

# from matchmaker.base import OnlineAlignment
# from matchmaker.dp.dtw_loop import oltw_arzt_loop
# from matchmaker.features.audio import FRAME_RATE, QUEUE_TIMEOUT, WINDOW_SIZE
# from matchmaker.utils import (
#     CYTHONIZED_METRICS_W_ARGUMENTS,
#     CYTHONIZED_METRICS_WO_ARGUMENTS,
#     distances,
# )
# from matchmaker.utils.distances import Metric, vdist
# from matchmaker.utils.misc import (
#     MatchmakerInvalidOptionError,
#     MatchmakerInvalidParameterTypeError,
#     RECVQueue,
#     set_latency_stats,
# )

# STEP_SIZE: int = 5
# START_WINDOW_SIZE: Union[float, int] = 0.25


# class OnlineTimeWarpingArzt(OnlineAlignment):
#     """
#     Fast On-line Time Warping

#     Parameters
#     ----------
#     reference_features : np.ndarray
#         A 2D array with dimensions (`n_timesteps`, `n_features`) containing the
#         features of the reference the input is going to be aligned to.

#     window_size : int
#         Size of the window for searching the optimal path in the cumulative
#         cost matrix.

#     step_size : int
#         Size of the step

#     distance_func : str, tuple (str, dict) or callable
#         Distance function for computing pairwise distances.

#     start_window_size: int
#         Size of the starting window size.

#     Attributes
#     ----------
#     reference_features : np.ndarray
#         See description above.

#     window_size : int
#         See description above.

#     step_size : int
#         See description above.

#     input_features : list
#         List with the input features (updates every time there is a step).

#     current_position : int
#         Index of the current position.

#     warping_path : list
#         List of tuples containing the current position and the corresponding
#         index in the array of `reference_features`.

#     positions : list
#         List of the positions for each input.
#     """

#     DEFAULT_DISTANCE_FUNC: str = "Manhattan"
#     distance_func: Callable[[NDArray[np.float32]], NDArray[np.float32]]
#     vdist: Callable[[NDArray[np.float32]], NDArray[np.float32]]

#     def __init__(
#         self,
#         reference_features: NDArray[np.float32],
#         window_size: int = WINDOW_SIZE,
#         step_size: int = STEP_SIZE,
#         distance_func: Union[
#             str,
#             Callable,
#             Tuple[str, Dict[str, Any]],
#         ] = DEFAULT_DISTANCE_FUNC,
#         start_window_size: int = START_WINDOW_SIZE,
#         current_position: int = 0,
#         frame_rate: int = FRAME_RATE,
#         queue: Optional[RECVQueue] = None,
#         **kwargs,
#     ) -> None:
#         super().__init__(reference_features=reference_features)

#         self.input_features: List[NDArray[np.float32]] = None
#         self.queue = queue or RECVQueue()

#         if not (isinstance(distance_func, (str, tuple)) or callable(distance_func)):
#             raise MatchmakerInvalidParameterTypeError(
#                 parameter_name="distance_func",
#                 required_parameter_type=(str, tuple, Callable),
#                 actual_parameter_type=type(distance_func),
#             )

#         # Set local cost function
#         if isinstance(distance_func, str):
#             if distance_func not in CYTHONIZED_METRICS_WO_ARGUMENTS:
#                 raise MatchmakerInvalidOptionError(
#                     parameter_name="distance_func",
#                     valid_options=CYTHONIZED_METRICS_WO_ARGUMENTS,
#                     value=distance_func,
#                 )
#             # If distance_func is a string
#             self.distance_func = getattr(distances, distance_func)()

#         elif isinstance(distance_func, tuple):
#             if distance_func[0] not in CYTHONIZED_METRICS_W_ARGUMENTS:
#                 raise MatchmakerInvalidOptionError(
#                     parameter_name="distance_func",
#                     valid_options=CYTHONIZED_METRICS_W_ARGUMENTS,
#                     value=distance_func[0],
#                 )
#             # distance_func is a tuple with the arguments to instantiate
#             # the cost
#             self.distance_func = getattr(distances, distance_func[0])(
#                 **distance_func[1]
#             )

#         elif callable(distance_func):
#             # If the local cost is a callable
#             self.distance_func = distance_func

#         # A callable to compute the distance between the rows of matrix and a vector
#         if isinstance(self.distance_func, Metric):
#             self.vdist = vdist
#         else:
#             # TODO: Speed this up somehow instead of list comprehension
#             self.vdist = lambda X, y, lcf: np.array([lcf(x, y) for x in X]).astype(
#                 np.float32
#             )

#         self.N_ref: int = self.reference_features.shape[0]
#         self.frame_rate = frame_rate
#         self.window_size: int = window_size * self.frame_rate
#         self.step_size: int = step_size
#         self.start_window_size: int = int(np.round(start_window_size * frame_rate))
#         self.init_position: int = current_position
#         self.current_position: int = current_position
#         self.positions: List[int] = []
#         self._warping_path: List = []
#         self.global_cost_matrix: NDArray[np.float32] = (
#             np.ones((reference_features.shape[0] + 1, 2), dtype=np.float32) * np.inf
#         ).astype(np.float32)
#         self.input_index: int = 0
#         self.go_backwards: bool = False
#         self.update_window_index: bool = False
#         self.restart: bool = False
#         self.is_following: bool = False
#         self.last_queue_update = time.time()
#         self.latency_stats: Dict[str, float] = {
#             "total_latency": 0,
#             "total_frames": 0,
#             "max_latency": 0,
#             "min_latency": float("inf"),
#         }

#     @property
#     def warping_path(self) -> NDArray[np.int32]:
#         wp = (np.array(self._warping_path).T).astype(np.int32)
#         return wp

#     def __call__(self, input: NDArray[np.float32]) -> int:
#         self.step(input)
#         return self.current_position

#     def run(self, verbose: bool = True) -> Generator[int, None, NDArray[np.float32]]:
#         """Run the online alignment process.

#         Parameters
#         ----------
#         verbose : bool, optional
#             Whether to show progress bar, by default True

#         Yields
#         ------
#         int
#             Current position in the reference sequence

#         Returns
#         -------
#         NDArray[np.float32]
#             The warping path as a 2D array where each column contains
#             (reference_position, input_position)
#         """
#         self.reset()

#         if verbose:
#             pbar = progressbar.ProgressBar(max_value=self.N_ref, redirect_stdout=True)

#         while self.is_still_following():
#             features, f_time = self.queue.get(timeout=QUEUE_TIMEOUT)
#             self.last_queue_update = time.time()
#             self.input_features = (
#                 np.concatenate((self.input_features, features))
#                 if self.input_features is not None
#                 else features
#             )
#             self.step(features)

#             if verbose:
#                 pbar.update(int(self.current_position))

#             latency = time.time() - self.last_queue_update
#             self.latency_stats = set_latency_stats(
#                 latency, self.latency_stats, self.input_index
#             )
#             yield self.current_position

#         if verbose:
#             pbar.finish()

#         return self.warping_path

#     def is_still_following(self):
#         # TODO: check stopping if the follower is stuck.
#         return self.current_position <= self.N_ref - 2

#     def reset(self) -> None:
#         self.current_position = self.init_position
#         self.positions = []
#         self._warping_path: List = []
#         self.global_cost_matrix = (
#             np.ones((self.reference_features.shape[0] + 1, 2), dtype=np.float32)
#             * np.inf
#         )
#         self.input_index = 0
#         self.update_window_index = False

#     def get_window(self) -> Tuple[int, int]:
#         w_size = self.window_size
#         if self.window_index < self.start_window_size:
#             w_size = self.start_window_size
#         window_start = max(self.window_index - w_size, 0)
#         window_end = min(self.window_index + w_size, self.N_ref)
#         return window_start, window_end

#     @property
#     def window_index(self) -> int:
#         return self.current_position

#     def step(self, input_features: NDArray[np.float32]) -> None:
#         """
#         Update the current position and the warping path.
#         """
#         min_costs = np.inf
#         min_index = max(self.window_index - self.step_size, 0)

#         window_start, window_end = self.get_window()
#         # compute local cost beforehand as it is much faster (~twice as fast)
#         window_cost = self.vdist(
#             self.reference_features[window_start:window_end],
#             input_features.squeeze(),
#             self.distance_func,
#         )

#         self.global_cost_matrix, min_index, min_costs = oltw_arzt_loop(
#             global_cost_matrix=self.global_cost_matrix,
#             window_cost=window_cost,
#             window_start=window_start,
#             window_end=window_end,
#             input_index=self.input_index,
#             min_costs=min_costs,
#             min_index=min_index,
#         )

#         # adapt current_position: do not go backwards,
#         # but also go a maximum of N steps forward

#         if self.input_index == 0:
#             # enforce the first time step to stay at the
#             # initial position
#             self.current_position = min(
#                 max(self.current_position, min_index),
#                 self.current_position,
#             )
#         else:
#             self.current_position = min(
#                 max(self.current_position, min_index),
#                 self.current_position + self.step_size,
#             )

#         self._warping_path.append((self.current_position, self.input_index))
#         # update input index
#         self.input_index += 1


# if __name__ == "__main__":
#     pass

#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
On-line Dynamic Time Warping with VIVACE Cognitive Layer
"""

import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import progressbar
from numpy.typing import NDArray

from matchmaker.base import OnlineAlignment
from matchmaker.dp.dtw_loop import oltw_arzt_loop
from matchmaker.features.audio import FRAME_RATE, QUEUE_TIMEOUT, WINDOW_SIZE
from matchmaker.utils import (
    CYTHONIZED_METRICS_W_ARGUMENTS,
    CYTHONIZED_METRICS_WO_ARGUMENTS,
    distances,
)
from matchmaker.utils.distances import Metric, vdist
from matchmaker.utils.misc import (
    MatchmakerInvalidOptionError,
    MatchmakerInvalidParameterTypeError,
    RECVQueue,
    set_latency_stats,
)

STEP_SIZE: int = 5
START_WINDOW_SIZE: Union[float, int] = 0.25


class OnlineTimeWarpingArzt(OnlineAlignment):
    """
    Fast On-line Time Warping with VIVACE extensions
    """

    DEFAULT_DISTANCE_FUNC: str = "Manhattan"
    distance_func: Callable[[NDArray[np.float32]], NDArray[np.float32]]
    vdist: Callable[[NDArray[np.float32]], NDArray[np.float32]]

    def __init__(
        self,
        reference_features: NDArray[np.float32],
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        distance_func: Union[
            str,
            Callable,
            Tuple[str, Dict[str, Any]],
        ] = DEFAULT_DISTANCE_FUNC,
        start_window_size: int = START_WINDOW_SIZE,
        current_position: int = 0,
        frame_rate: int = FRAME_RATE,
        queue: Optional[RECVQueue] = None,
        # [VIVACE] Add Performance Plan argument
        performance_plan=None,
        tempo_bpm: float = 120.0,  # [VIVACE] BPM for frame→beat conversion
        is_gap_frame: Optional[np.ndarray] = None,  # [VIVACE] MIDI-derived gap mask
        normalization: str = "arzt",  # [SCORPION] "arzt" or "scorpion"
        **kwargs,
    ) -> None:
        super().__init__(reference_features=reference_features)

        self.input_features: List[NDArray[np.float32]] = None
        self.queue = queue or RECVQueue()

        # [VIVACE] Store the brain
        self.performance_plan = performance_plan
        self.tempo_bpm = tempo_bpm  # [VIVACE] For frame→beat conversion
        self.is_gap_frame = is_gap_frame  # [VIVACE] Gap mask for silence robustness
        self.normalization = normalization  # [SCORPION] "arzt" or "scorpion"
        self.norm_exponent = kwargs.get('norm_exponent', 0.5)  # [SCORPION] tunable

        if not (isinstance(distance_func, (str, tuple)) or callable(distance_func)):
            raise MatchmakerInvalidParameterTypeError(
                parameter_name="distance_func",
                required_parameter_type=(str, tuple, Callable),
                actual_parameter_type=type(distance_func),
            )

        # Set local cost function
        if isinstance(distance_func, str):
            if distance_func not in CYTHONIZED_METRICS_WO_ARGUMENTS:
                raise MatchmakerInvalidOptionError(
                    parameter_name="distance_func",
                    valid_options=CYTHONIZED_METRICS_WO_ARGUMENTS,
                    value=distance_func,
                )
            # If distance_func is a string
            self.distance_func = getattr(distances, distance_func)()

        elif isinstance(distance_func, tuple):
            if distance_func[0] not in CYTHONIZED_METRICS_W_ARGUMENTS:
                raise MatchmakerInvalidOptionError(
                    parameter_name="distance_func",
                    valid_options=CYTHONIZED_METRICS_W_ARGUMENTS,
                    value=distance_func[0],
                )
            # distance_func is a tuple with the arguments to instantiate
            # the cost
            self.distance_func = getattr(distances, distance_func[0])(
                **distance_func[1]
            )

        elif callable(distance_func):
            # If the local cost is a callable
            self.distance_func = distance_func

        # A callable to compute the distance between the rows of matrix and a vector
        if isinstance(self.distance_func, Metric):
            self.vdist = vdist
        else:
            # TODO: Speed this up somehow instead of list comprehension
            self.vdist = lambda X, y, lcf: np.array([lcf(x, y) for x in X]).astype(
                np.float32
            )

        self.N_ref: int = self.reference_features.shape[0]
        self.frame_rate = frame_rate
        self.window_size: int = window_size * self.frame_rate
        self.step_size: int = step_size
        self.start_window_size: int = int(np.round(start_window_size * frame_rate))
        self.init_position: int = current_position
        self.current_position: int = current_position
        self.positions: List[int] = []
        self._warping_path: List = []
        self.global_cost_matrix: NDArray[np.float32] = (
            np.ones((reference_features.shape[0] + 1, 2), dtype=np.float32) * np.inf
        ).astype(np.float32)
        # [SCORPION] Path-length matrix for path-length normalization
        if self.normalization == "scorpion":
            self.path_length_matrix: NDArray[np.float32] = np.zeros(
                (reference_features.shape[0] + 1, 2), dtype=np.float32
            )
        self.input_index: int = 0
        self.go_backwards: bool = False
        self.update_window_index: bool = False
        self.restart: bool = False
        self.is_following: bool = False
        self.last_queue_update = time.time()
        self.latency_stats: Dict[str, float] = {
            "total_latency": 0,
            "total_frames": 0,
            "max_latency": 0,
            "min_latency": float("inf"),
        }

    @property
    def warping_path(self) -> NDArray[np.int32]:
        wp = (np.array(self._warping_path).T).astype(np.int32)
        return wp

    def __call__(self, input: NDArray[np.float32]) -> int:
        self.step(input)
        return self.current_position

    def run(self, verbose: bool = True) -> Generator[int, None, NDArray[np.float32]]:
        """Run the online alignment process."""
        self.reset()

        if verbose:
            pbar = progressbar.ProgressBar(max_value=self.N_ref, redirect_stdout=True)

        while self.is_still_following():
            item = self.queue.get(timeout=QUEUE_TIMEOUT)
            if item is None:
                # End-of-stream sentinel: stop following
                self.current_position = self.N_ref
                break
            features, f_time = item
            self.last_queue_update = time.time()
            self.input_features = (
                np.concatenate((self.input_features, features))
                if self.input_features is not None
                else features
            )
            self.step(features)

            if verbose:
                pbar.update(int(self.current_position))

            latency = time.time() - self.last_queue_update
            self.latency_stats = set_latency_stats(
                latency, self.latency_stats, self.input_index
            )
            yield self.current_position

        # [VIVACE] Save memory at the end of run
        if self.performance_plan:
            self.performance_plan.save_plan()

        if verbose:
            pbar.finish()

        return self.warping_path

    def is_still_following(self):
        # TODO: check stopping if the follower is stuck.
        return self.current_position <= self.N_ref - 2

    def reset(self) -> None:
        self.current_position = self.init_position
        self.positions = []
        self._warping_path: List = []
        self.global_cost_matrix = (
            np.ones((self.reference_features.shape[0] + 1, 2), dtype=np.float32)
            * np.inf
        )
        self.input_index = 0
        self.update_window_index = False

    def get_window(self) -> Tuple[int, int]:
        w_size = self.window_size
        if self.window_index < self.start_window_size:
            w_size = self.start_window_size

        # [SCORPION] Adaptive window: expand at beats where previous
        # rehearsals had large errors (stored in PerformancePlan)
        if self.performance_plan is not None and self.tempo_bpm:
            current_beat = self.current_position / self.frame_rate * (self.tempo_bpm / 60.0)
            adaptive_w = self.performance_plan.get_window_size(current_beat)
            if adaptive_w is not None:
                w_size = max(w_size, int(adaptive_w * self.frame_rate))

        window_start = max(self.window_index - w_size, 0)
        window_end = min(self.window_index + w_size, self.N_ref)
        return window_start, window_end

    @property
    def window_index(self) -> int:
        return self.current_position

    def step(self, input_features: NDArray[np.float32]) -> None:
        """
        Update the current position and the warping path with VIVACE adaptive logic.
        """
        min_costs = np.inf
        # Note: The search window lower bound logic remains the same
        min_index = max(self.window_index - self.step_size, 0)

        window_start, window_end = self.get_window()
        # compute local cost beforehand as it is much faster (~twice as fast)
        window_cost = self.vdist(
            self.reference_features[window_start:window_end],
            input_features.squeeze(),
            self.distance_func,
        )

        # [SCORPION-OLTW] Always use standard Arzt (i+j+1) normalization.
        # The former dampened (i+j+1)^alpha variant was a known negative result
        # (never matched Dixon). The real SCORPION-OLTW contribution is the
        # silent-beat freeze applied at the position-update layer below.
        self.global_cost_matrix, min_index, min_costs = oltw_arzt_loop(
            global_cost_matrix=self.global_cost_matrix,
            window_cost=window_cost,
            window_start=window_start,
            window_end=window_end,
            input_index=self.input_index,
            min_costs=min_costs,
            min_index=min_index,
        )

        # =================================================================
        # [VIVACE] Active Listening Logic (Equations 1, 2, 3)
        # =================================================================
        
        # 1. Calculate Tracking Confidence C(t)
        # We use the minimum local cost as a proxy. Lower cost = Higher match.
        # Normalize to 0-1 range using exponential decay.
        current_min_cost = np.min(window_cost) if len(window_cost) > 0 else 100.0
        confidence = np.exp(-current_min_cost)

        # 2. Adaptive Step Size Calculation
        active_step_limit = self.step_size # Default

        if self.performance_plan:
            # Retrieve expected tempo ratio (gamma) from memory
            # Convert frame index → beat index for plan lookup
            current_beat = self.current_position / self.frame_rate * (self.tempo_bpm / 60.0)
            gamma, _ = self.performance_plan.get_params(current_beat)

            if gamma < 0.5:
                # [VIVACE] Explicit intent: force 1:1 advancement at gaps.
                min_index = self.current_position + 1
                active_step_limit = 1
            elif self.performance_plan.is_locked:
                # [SCORPION] Injected intent: bypass confidence gating.
                # When intent is explicitly injected (not rehearsal-learned),
                # trust the plan directly without VIVACE's sigma gate.
                active_step_limit = int(np.clip(np.round(self.step_size * gamma), 1, 10))
            else:
                # Eq 3: Sigmoid gating function (Soft Gate)
                # k=10, theta=0.5 (from paper)
                sigma = 1 / (1 + np.exp(-10 * (confidence - 0.5)))

                # Eq 2: Active step size based on memory
                delta_active = np.clip(np.floor(self.step_size * gamma), 1, 10)

                # Eq 1: Fused step size
                # If confident (sigma -> 1): use active step (memory)
                # If uncertain (sigma -> 0): use default step (safe)
                adaptive_step = sigma * delta_active + (1 - sigma) * self.step_size
                active_step_limit = int(np.round(adaptive_step))
        
        # =================================================================

        # [VIVACE] Gap-robust override: use MIDI-derived gap mask to detect
        # silence/rest regions where normalization bias causes overshoot.
        # During gaps, force 1:1 advancement to prevent DTW drift.
        if self.is_gap_frame is not None and self.current_position < len(self.is_gap_frame):
            if self.is_gap_frame[self.current_position]:
                min_index = self.current_position + 1
                active_step_limit = 1

        # adapt current_position: do not go backwards,
        # but also go a maximum of N steps forward

        prev_position = self.current_position

        if self.input_index == 0:
            # enforce the first time step to stay at the initial position
            self.current_position = min(
                max(self.current_position, min_index),
                self.current_position,
            )
        else:
            # [VIVACE] Use active_step_limit instead of fixed self.step_size
            self.current_position = min(
                max(self.current_position, min_index),
                self.current_position + active_step_limit,
            )

        # [VIVACE] Stuck detection & recovery.
        if self.current_position == prev_position:
            self._stuck_count = getattr(self, '_stuck_count', 0) + 1
        else:
            self._stuck_count = 0

        if self._stuck_count > self.frame_rate:  # stuck for >1 second
            self.current_position = prev_position + 1
            self._stuck_count = 0


        self._warping_path.append((self.current_position, self.input_index))

        # =================================================================
        # [VIVACE] Performance Plan Update
        # NOTE: Frame-by-frame WEMA update is disabled — too noisy
        # (instant_tempo_ratio is 0-5 integer, causes plan degradation).
        # Instead, use post-hoc update via update_plan_from_warping_path()
        # after the DTW run completes.
        # =================================================================
        # =================================================================

        # update input index
        self.input_index += 1


if __name__ == "__main__":
    pass