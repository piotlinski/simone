# Note: These constants can't necessarily be changed without modifying the code elsewhere.

# Architecture parameters
ENCODER_CONV_CHANNELS = 128
TRANSFORMER_CHANNELS = 320
LATENT_CHANNELS = 32
# The number of "object" latents
K = 16
# The number of frames in the input video sequence
T = 16
XY_RESOLUTION = 64
CONV_LAYERS = 3
CONV_STRIDE = 2
XY_SPATIAL_DIM_AFTER_CONV_ENCODER = XY_RESOLUTION // CONV_STRIDE**CONV_LAYERS
XY_SPATIAL_DIM_AFTER_TRANSFORMER = XY_SPATIAL_DIM_AFTER_CONV_ENCODER // 2

# This spatial shape gets reinterpreted as K in the model
assert XY_SPATIAL_DIM_AFTER_TRANSFORMER**2 == K


# Base logging frequency (in steps)
LOG_FREQ = 500
