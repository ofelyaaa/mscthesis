from pydub import AudioSegment
import numpy as np
import soundfile as sf
import scipy.signal as signal
import os
from scipy.signal import convolve
import simpleaudio as sa

os.chdir("/Users/ofelyaaliyeva/Desktop")

sound = AudioSegment.from_wav("ClassroomNoise.wav")
samples = np.array(sound.get_array_of_samples()).astype(np.float32)
sample_rate = sound.frame_rate

# Normalize to [-1.0, 1.0]
samples /= 2 ** 15


# --- Low-pass filter to reduce high-frequency reverberation (higher cutoff)
def lowpass_filter(signal_array, sr, cutoff_hz=8000):  # Higher cutoff to preserve more high freq
    nyquist = 0.5 * sr
    normal_cutoff = cutoff_hz / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, signal_array)


# --- Reduce reverb by gently damping delayed signals (very light decay)
def reduce_reverb_light(signal_array, sr, decay=0.1, delay_ms=20):  # Very light decay and short delay
    delay_samples = int(sr * delay_ms / 1000)
    impulse = np.zeros(delay_samples + 1)
    impulse[0] = 1
    impulse[-1] = decay  # Extremely light decay
    convolved_signal = convolve(signal_array, impulse, mode='same')

    # Low-pass filter to reduce high-frequency reverberation (preserve more clarity)
    filtered_signal = lowpass_filter(convolved_signal, sr)
    return filtered_signal


# --- Normalize to target RMS (e.g., 0.02 ≈ 65 dB SPL)
def normalize_rms(signal_array, target_rms=0.02):
    current_rms = np.sqrt(np.mean(signal_array ** 2))
    if current_rms == 0:
        return signal_array  # avoid division by zero
    scaling_factor = target_rms / current_rms
    return signal_array * scaling_factor


# --- Process audio to reduce reverb lightly and normalize volume
processed = reduce_reverb_light(samples, sample_rate)
processed = normalize_rms(processed, target_rms=0.02)  # Adjust for 65 dB SPL

# Confirm new RMS and SPL
rms = np.sqrt(np.mean(processed ** 2))
db = 20 * np.log10(rms) if rms > 0 else -100

print(f"Adjusted Amplitude (RMS): {rms:.4f}")
print(f"Approx dB SPL: {db + 100:.1f} dB")  # Should be close to 65 dB

# Clip values just in case before saving
processed = np.clip(processed, -1.0, 1.0)
processed_int16 = np.int16(processed * 32767)

# --- Play the audio before saving (using simpleaudio)
# Convert processed audio back to pydub AudioSegment format for playback
processed_sound = AudioSegment(
    processed_int16.tobytes(),
    frame_rate=sample_rate,
    sample_width=processed_int16.itemsize,
    channels=1
)

# Play the audio
play_obj = sa.play_buffer(processed_int16, 1, 2, sample_rate)
#play_obj.wait_done()  # Wait until the playback is finished

# Save processed audio
sf.write("highintensity_lowreverb.wav", processed_int16, sample_rate)
print("Processed file saved.")
