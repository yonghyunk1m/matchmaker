# #!/usr/bin/python
# # -*- coding: utf-8 -*-
# """
# Input MIDI stream
# """

# import time
# from types import TracebackType
# from typing import Callable, List, Optional, Tuple, Type, Union

# import mido
# from mido.ports import BaseInput as MidiInputPort

# from matchmaker.features.midi import PitchIOIProcessor
# from matchmaker.io.mediator import CeusMediator
# from matchmaker.utils.misc import RECVQueue
# from matchmaker.utils.processor import Processor
# from matchmaker.utils.stream import Stream
# from matchmaker.utils.symbolic import (
#     Buffer,
#     framed_midi_messages_from_performance,
#     get_available_midi_port,
#     midi_messages_from_performance,
# )

# # Default polling period (in seconds)
# POLLING_PERIOD = 0.01


# class MidiStream(Stream):
#     """
#     A class to process input MIDI stream in real time

#     Parameters
#     ----------
#     port : mido.ports.BaseInput
#         Input MIDI port

#     queue : RECVQueue
#         Queue to store processed MIDI input

#     init_time : Optional[float]
#         The initial time. If none given, the
#         initial time will be set to the starting time
#         of the thread.

#     return_midi_messages: bool
#         Return MIDI messages in addition to the
#         processed features.

#     mediator : CeusMediator or None
#         A Mediator instance to filter input MIDI.
#         This is useful for certain older instruments,
#         like the Bösendorfer CEUS, which do not distinguish
#         between notes played by a human, and notes sent
#         from a different process  (e.g., an accompaniment system)
#     """

#     midi_in: Optional[MidiInputPort]
#     init_time: float
#     listen: bool
#     queue: RECVQueue
#     processor: Callable
#     return_midi_messages: bool
#     first_message: bool
#     mediator: CeusMediator
#     is_windowed: bool
#     polling_period: Optional[float]
#     midi_messages: List[Tuple[mido.Message, float]]

#     def __init__(
#         self,
#         processor: Optional[Union[Callable, Processor]] = None,
#         file_path: Optional[str] = None,
#         polling_period: Optional[float] = POLLING_PERIOD,
#         port: Optional[Union[MidiInputPort, str]] = None,
#         queue: RECVQueue = None,
#         init_time: Optional[float] = None,
#         return_midi_messages: bool = False,
#         mediator: Optional[CeusMediator] = None,
#         virtual_port: bool = False,
#     ):
#         if processor is None:
#             processor = PitchIOIProcessor()

#         Stream.__init__(
#             self,
#             processor=processor,
#             mock=file_path is not None,
#         )
#         self.file_path = file_path

#         if isinstance(port, str) or port is None and file_path is None:
#             port_name = get_available_midi_port(port, is_virtual=virtual_port)
#             self.midi_in = mido.open_input(port_name, virtual=virtual_port)
#         elif isinstance(port, MidiInputPort) and file_path is None:
#             self.midi_in = port
#         else:
#             self.midi_in = None

#         self.init_time = init_time
#         self.listen = False
#         self.queue = queue or RECVQueue()
#         self.first_msg = False
#         self.return_midi_messages = return_midi_messages
#         self.mediator = mediator
#         self.midi_messages = []

#         self.polling_period = polling_period
#         if (polling_period is None) and (self.mock is False):
#             self.is_windowed = False
#             self.run = self.run_online_single
#             self._process_frame = self._process_frame_message

#         elif (polling_period is None) and (self.mock is True):
#             self.is_windowed = False
#             self.run = self.run_offline_single
#             self._process_frame = self._process_frame_message

#         elif (polling_period is not None) and (self.mock is False):
#             self.is_windowed = True
#             self.run = self.run_online_windowed
#             self._process_frame = self._process_frame_window

#         elif (polling_period is not None) and (self.mock is True):
#             self.is_windowed = True
#             self.run = self.run_offline_windowed
#             self._process_frame = self._process_frame_window

#     def _process_frame_message(
#         self,
#         data: mido.Message,
#         *args,
#         c_time: float,
#         **kwargs,
#     ) -> None:
#         output = self.processor(([(data, c_time)], c_time))
#         if self.return_midi_messages:
#             self.queue.put(((data, c_time), output))
#         else:
#             self.queue.put(output)

#     def _process_frame_window(
#         self,
#         data: Buffer,
#         *args,
#         **kwargs,
#     ) -> None:
#         # the data is the Buffer instance
#         output = self.processor((data.frame[:], data.time))

#         # if output is not None:
#         if self.return_midi_messages:
#             self.queue.put((data.frame, output))
#         else:
#             self.queue.put(output)

#     def run_online_single(self):
#         self.start_listening()
#         while self.listen:
#             msg = self.midi_in.poll()
#             if msg is not None:
#                 if (
#                     self.mediator is not None
#                     and msg.type == "note_on"
#                     and self.mediator.filter_check(msg.note)
#                 ):
#                     continue
#                 c_time = self.current_time
#                 self.add_midi_message(
#                     msg=msg,
#                     time=c_time,
#                 )
#                 self._process_frame_message(
#                     data=msg,
#                     c_time=c_time,
#                 )

#     def run_online_windowed(self):
#         """ """
#         self.start_listening()
#         frame = Buffer(self.polling_period)
#         frame.start = self.current_time

#         # TODO: check the effect of smaller st
#         st = self.polling_period * 0.001
#         while self.listen:
#             time.sleep(st)
#             if self.listen:
#                 # added if to check once again after sleep
#                 c_time = self.current_time
#                 msg = self.midi_in.poll()
#                 if msg is not None:
#                     if (
#                         self.mediator is not None
#                         and (msg.type == "note_on" and msg.velocity > 0)
#                         and self.mediator.filter_check(msg.note)
#                     ):
#                         continue
#                     self.add_midi_message(
#                         msg=msg,
#                         time=c_time,
#                     )
#                     if msg.type in ["note_on", "note_off"]:
#                         # TODO: check changing self.current_time for c_time
#                         frame.append(msg, c_time)
#                         if not self.first_msg:
#                             self.first_msg = True

#                 if c_time >= frame.end and self.first_msg:
#                     self._process_frame_window(data=frame)
#                     frame.reset(c_time)

#     def run_offline_single(self):
#         """
#         Simulate real-time stream as loop iterating
#         over MIDI messages
#         """
#         midi_messages, message_times = midi_messages_from_performance(
#             perf=self.file_path,
#         )
#         self.init_time = message_times.min()
#         self.start_listening()
#         for msg, c_time in zip(midi_messages, message_times):
#             self.add_midi_message(
#                 msg=msg,
#                 time=c_time,
#             )
#             self._process_frame_message(
#                 data=msg,
#                 c_time=c_time,
#             )
#         self.stop_listening()

#     def run_offline_windowed(self):
#         """
#         Simulate real-time stream as loop iterating
#         over MIDI messages
#         """
#         self.start_listening()
#         midi_frames, frame_times = framed_midi_messages_from_performance(
#             perf=self.file_path,
#             polling_period=self.polling_period,
#         )
#         self.init_time = frame_times.min()
#         for frame in midi_frames:
#             self._process_frame_window(
#                 data=frame,
#             )

#     @property
#     def current_time(self) -> Optional[float]:
#         """
#         Get current time since starting to listen
#         """
#         if self.init_time is None:
#             # TODO: Check if this has weird consequences
#             self.init_time = time.time()
#             return 0

#         return time.time() - self.init_time

#         # return time.time() - self.init_time if self.init_time is not None else None

#     def start_listening(self):
#         """Start listening to midi input (open input port and get starting time)"""
#         self.listen = True
#         if self.mock:
#             print("* Mock listening to stream....")
#         else:
#             print("* Start listening to MIDI stream....")
#         # set initial time
#         self.current_time

#     def stop_listening(self):
#         """Stop listening to MIDI input"""
#         if self.listen:
#             print("* Stop listening to MIDI stream....")
#         # break while loop in self.run
#         self.listen = False
#         # reset init time
#         self.init_time = None

#         if self.midi_in is not None:
#             self.midi_in.close()

#     def __enter__(self) -> None:
#         self.start()
#         return self

#     def __exit__(
#         self,
#         exc_type: Optional[Type[BaseException]],
#         exc_value: Optional[BaseException],
#         traceback: Optional[TracebackType],
#     ) -> Optional[bool]:
#         self.stop()
#         if exc_type is not None:  # pragma: no cover
#             # Returning True will suppress the exception
#             # False means the exception will propagate
#             return False
#         return True

#     def stop(self):
#         self.stop_listening()
#         self.join()

#     def clear_queue(self):
#         if self.queue.not_empty:
#             self.queue.queue.clear()

#     def add_midi_message(self, msg: mido.Message, time: float) -> None:
#         self.midi_messages.append((msg, time))



#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Input MIDI stream
"""

import time
from types import TracebackType
from typing import Callable, List, Optional, Tuple, Type, Union

import mido
from mido.ports import BaseInput as MidiInputPort

from matchmaker.features.midi import PitchIOIProcessor
from matchmaker.io.mediator import CeusMediator
from matchmaker.utils.misc import RECVQueue
from matchmaker.utils.processor import Processor
from matchmaker.utils.stream import Stream
from matchmaker.utils.symbolic import (
    Buffer,
    framed_midi_messages_from_performance,
    get_available_midi_port,
    midi_messages_from_performance,
)

# Default polling period (in seconds)
POLLING_PERIOD = 0.01


class MidiStream(Stream):
    """
    A class to process input MIDI stream in real time
    """

    midi_in: Optional[MidiInputPort]
    init_time: float
    listen: bool
    queue: RECVQueue
    processor: Callable
    return_midi_messages: bool
    first_message: bool
    mediator: CeusMediator
    is_windowed: bool
    polling_period: Optional[float]
    midi_messages: List[Tuple[mido.Message, float]]

    def __init__(
        self,
        processor: Optional[Union[Callable, Processor]] = None,
        file_path: Optional[str] = None,
        polling_period: Optional[float] = POLLING_PERIOD,
        port: Optional[Union[MidiInputPort, str]] = None,
        queue: RECVQueue = None,
        init_time: Optional[float] = None,
        return_midi_messages: bool = False,
        mediator: Optional[CeusMediator] = None,
        virtual_port: bool = False,
    ):
        if processor is None:
            processor = PitchIOIProcessor()

        Stream.__init__(
            self,
            processor=processor,
            mock=file_path is not None,
        )
        self.file_path = file_path

        if isinstance(port, str) or port is None and file_path is None:
            port_name = get_available_midi_port(port, is_virtual=virtual_port)
            self.midi_in = mido.open_input(port_name, virtual=virtual_port)
        elif isinstance(port, MidiInputPort) and file_path is None:
            self.midi_in = port
        else:
            self.midi_in = None

        self.init_time = init_time
        self.listen = False
        self.queue = queue or RECVQueue()
        self.first_msg = False
        self.return_midi_messages = return_midi_messages
        self.mediator = mediator
        self.midi_messages = []

        self.polling_period = polling_period
        if (polling_period is None) and (self.mock is False):
            self.is_windowed = False
            self.run = self.run_online_single
            self._process_frame = self._process_frame_message

        elif (polling_period is None) and (self.mock is True):
            self.is_windowed = False
            self.run = self.run_offline_single
            self._process_frame = self._process_frame_message

        elif (polling_period is not None) and (self.mock is False):
            self.is_windowed = True
            self.run = self.run_online_windowed
            self._process_frame = self._process_frame_window

        elif (polling_period is not None) and (self.mock is True):
            self.is_windowed = True
            self.run = self.run_offline_windowed
            self._process_frame = self._process_frame_window

    def _process_frame_message(
        self,
        data: mido.Message,
        *args,
        c_time: float,
        **kwargs,
    ) -> None:
        output = self.processor(([(data, c_time)], c_time))
        if self.return_midi_messages:
            self.queue.put(((data, c_time), output))
        else:
            self.queue.put(output)

    def _process_frame_window(
        self,
        data: Buffer,
        *args,
        **kwargs,
    ) -> None:
        # the data is the Buffer instance
        output = self.processor((data.frame[:], data.time))

        # if output is not None:
        if self.return_midi_messages:
            self.queue.put((data.frame, output))
        else:
            self.queue.put(output)

    def run_online_single(self):
        self.start_listening()
        while self.listen:
            msg = self.midi_in.poll()
            if msg is not None:
                if (
                    self.mediator is not None
                    and msg.type == "note_on"
                    and self.mediator.filter_check(msg.note)
                ):
                    continue
                c_time = self.current_time
                self.add_midi_message(
                    msg=msg,
                    time=c_time,
                )
                self._process_frame_message(
                    data=msg,
                    c_time=c_time,
                )

    def run_online_windowed(self):
        """ """
        self.start_listening()
        frame = Buffer(self.polling_period)
        frame.start = self.current_time

        # TODO: check the effect of smaller st
        st = self.polling_period * 0.001
        while self.listen:
            time.sleep(st)
            if self.listen:
                # added if to check once again after sleep
                c_time = self.current_time
                msg = self.midi_in.poll()
                if msg is not None:
                    if (
                        self.mediator is not None
                        and (msg.type == "note_on" and msg.velocity > 0)
                        and self.mediator.filter_check(msg.note)
                    ):
                        continue
                    self.add_midi_message(
                        msg=msg,
                        time=c_time,
                    )
                    if msg.type in ["note_on", "note_off"]:
                        # TODO: check changing self.current_time for c_time
                        frame.append(msg, c_time)
                        if not self.first_msg:
                            self.first_msg = True

                if c_time >= frame.end and self.first_msg:
                    self._process_frame_window(data=frame)
                    frame.reset(c_time)

    # def run_offline_single(self):
    #     """
    #     Simulate real-time stream as loop iterating
    #     over MIDI messages
    #     """
    #     midi_messages, message_times = midi_messages_from_performance(
    #         perf=self.file_path,
    #     )
    #     self.init_time = message_times.min()
    #     self.start_listening()
    #     for msg, c_time in zip(midi_messages, message_times):
    #         self.add_midi_message(
    #             msg=msg,
    #             time=c_time,
    #         )
    #         self._process_frame_message(
    #             data=msg,
    #             c_time=c_time,
    #         )
    #     self.stop_listening()

    # def run_offline_windowed(self):
    #     """
    #     Simulate real-time stream as loop iterating
    #     over MIDI messages
    #     """
    #     self.start_listening()
    #     midi_frames, frame_times = framed_midi_messages_from_performance(
    #         perf=self.file_path,
    #         polling_period=self.polling_period,
    #     )
    #     self.init_time = frame_times.min()
    #     for frame in midi_frames:
    #         # [VIVACE FIX] 버퍼에 있는 메시지를 히스토리에도 저장해야 합니다!
    #         for msg, t in frame.frame:
    #             self.add_midi_message(msg, t)
                
    #         self._process_frame_window(
    #             data=frame,
    #         )

    def run_offline_single(self):
        """
        Simulate real-time stream as loop iterating over MIDI messages
        [VIVACE FIX] Added time.sleep to simulate real-time playback
        """
        midi_messages, message_times = midi_messages_from_performance(
            perf=self.file_path,
        )
        
        # 파일 내의 시작 시간과 실제 시뮬레이션 시작 시간 동기화
        if len(message_times) > 0:
            file_start_time = message_times.min()
        else:
            file_start_time = 0.0
            
        real_start_time = time.time()
        self.init_time = real_start_time # 현재 시간을 init_time으로 설정
        
        self.start_listening()
        
        for msg, c_time in zip(midi_messages, message_times):
            # 1. 이 메시지가 언제 연주되어야 하는지 계산 (Offset)
            target_offset = c_time - file_start_time
            
            # 2. 실제 얼마나 시간이 흘렀는지 확인
            current_elapsed = time.time() - real_start_time
            
            # 3. 아직 시간이 안 됐다면 대기 (핵심!)
            wait_time = target_offset - current_elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            
            # 4. 데이터 처리
            # (시뮬레이션 중인 현재 시간 계산)
            simulated_time = time.time() - real_start_time
            
            self.add_midi_message(
                msg=msg,
                time=simulated_time,
            )
            self._process_frame_message(
                data=msg,
                c_time=simulated_time,
            )
        self.stop_listening()

    # def run_offline_windowed(self):
    #     """
    #     Simulate real-time stream as loop iterating over MIDI messages
    #     [VIVACE FIX] Added time.sleep to simulate real-time playback
    #     """
    #     self.start_listening()
    #     midi_frames, frame_times = framed_midi_messages_from_performance(
    #         perf=self.file_path,
    #         polling_period=self.polling_period,
    #     )
        
    #     # 파일 내의 시작 시간과 실제 시뮬레이션 시작 시간 동기화
    #     if len(frame_times) > 0:
    #         file_start_time = frame_times.min()
    #     else:
    #         file_start_time = 0.0

    #     real_start_time = time.time()
    #     self.init_time = real_start_time
        
    #     # frame과 frame_times를 함께 순회
    #     for frame, f_time in zip(midi_frames, frame_times):
            
    #         # 1. 이 프레임이 언제 처리되어야 하는지 계산
    #         target_offset = f_time - file_start_time
            
    #         # 2. 실제 경과 시간
    #         current_elapsed = time.time() - real_start_time
            
    #         # 3. 대기 (Sleep)
    #         wait_time = target_offset - current_elapsed
    #         if wait_time > 0:
    #             time.sleep(wait_time)
            
    #         # 4. 데이터 처리
    #         simulated_time = time.time() - real_start_time
            
    #         # [VIVACE] 버퍼 기록
    #         for msg, t in frame.frame:
    #             # 프레임 내부 시간도 상대 시간으로 보정해서 저장
    #             msg_offset = t - file_start_time
    #             self.add_midi_message(msg, msg_offset)
                
    #         self._process_frame_window(
    #             data=frame,
    #         )
    #     self.stop_listening()
    
    def run_offline_windowed(self):
        """
        Simulate real-time stream as loop iterating over MIDI messages
        [VIVACE FIX] Added time.sleep to simulate real-time playback
        """
        print("DEBUG: [VIVACE] run_offline_windowed is running! (Sleep Check)")
        self.start_listening()
        midi_frames, frame_times = framed_midi_messages_from_performance(
            perf=self.file_path,
            polling_period=self.polling_period,
        )
        
        # 파일 시작 시간 보정
        if len(frame_times) > 0:
            file_start_time = frame_times.min()
        else:
            file_start_time = 0.0

        # [중요] 현실 시간 기준점 잡기
        real_start_time = time.time()
        self.init_time = real_start_time
        
        for frame, f_time in zip(midi_frames, frame_times):
            # 1. 목표 시간 계산
            target_offset = f_time - file_start_time
            
            # 2. 현재 흐른 시간 계산
            current_elapsed = time.time() - real_start_time
            
            # 3. 박자 맞추기 (기다리기)
            wait_time = target_offset - current_elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            
            # 4. 데이터 처리 (저장용 시간은 시뮬레이션 시간 기준)
            simulated_time = time.time() - real_start_time
            
            # 버퍼에 기록 (이게 있어야 나중에 저장됨!)
            for msg, t in frame.frame:
                self.add_midi_message(msg, simulated_time)
                
            self._process_frame_window(
                data=frame,
            )
            
        self.stop_listening()

    @property
    def current_time(self) -> Optional[float]:
        """
        Get current time since starting to listen
        """
        if self.init_time is None:
            # TODO: Check if this has weird consequences
            self.init_time = time.time()
            return 0

        return time.time() - self.init_time

        # return time.time() - self.init_time if self.init_time is not None else None

    def start_listening(self):
        """Start listening to midi input (open input port and get starting time)"""
        self.listen = True
        if self.mock:
            print("* Mock listening to stream....")
        else:
            print("* Start listening to MIDI stream....")
        # set initial time
        self.current_time

    def stop_listening(self):
        """Stop listening to MIDI input"""
        if self.listen:
            print("* Stop listening to MIDI stream....")
        # break while loop in self.run
        self.listen = False
        # reset init time
        self.init_time = None

        if self.midi_in is not None:
            self.midi_in.close()

    def __enter__(self) -> None:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        self.stop()
        if exc_type is not None:  # pragma: no cover
            # Returning True will suppress the exception
            # False means the exception will propagate
            return False
        return True

    def stop(self):
        self.stop_listening()
        self.join()

    def clear_queue(self):
        if self.queue.not_empty:
            self.queue.queue.clear()

    # def add_midi_message(self, msg: mido.Message, time: float) -> None:
    #     self.midi_messages.append((msg, time))

    def add_midi_message(self, msg: mido.Message, time: float) -> None:
        self.midi_messages.append((msg, time))
        if len(self.midi_messages) % 100 == 0:
            print(f"[DEBUG] Captured {len(self.midi_messages)} messages...")
            
    # [VIVACE ADDITION] Start
    def start_recording(self):
        """Clear the buffer to start a clean recording"""
        self.midi_messages = []
        
    @property
    def recorded_messages(self):
        """
        Convert stored (msg, abs_time) tuples to a list of mido Messages 
        with delta-time (ticks) suitable for saving to a MIDI file.
        """
        if not self.midi_messages:
            return []
            
        TICKS_PER_SEC = 960 
        
        processed_msgs = []
        if self.midi_messages:
            last_time = self.midi_messages[0][1] 
            
            for msg, t in self.midi_messages:
                dt_sec = t - last_time
                if dt_sec < 0: dt_sec = 0
                
                dt_ticks = int(dt_sec * TICKS_PER_SEC)
                
                new_msg = msg.copy()
                new_msg.time = dt_ticks
                processed_msgs.append(new_msg)
                
                last_time = t
            
        return processed_msgs
    # [VIVACE ADDITION] End