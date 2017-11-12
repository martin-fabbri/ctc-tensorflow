window_size_ms=25 (recommended in literature for speech recognition) ->
   > How wide the input window is in samples. For the highest
     efficiency this should be a power of two,
     but other values are accepted.
   > python_speech_features also defaults window_size to 25


window_stride_ms=10 (recommended in literature for speech recognition) ->
   > How widely apart the center of adjacent sample windows
     should be.
   > python_speech_features also defaults winstep to 10

magnitude_squared=False (Default) -> todo: should we change this parameter to True?
   > Whether to return the squared magnitude or just the
     magnitude. Using squared magnitude can avoid extra
     calculations.
   > magnitude_squared=True -> seems to delay the network conversion

dct_coefficient_count=13 (default) ->
   > How many output channels to produce per time slice.
   > python_speech_features' numcep default is 13