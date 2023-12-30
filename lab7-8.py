import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from datetime import datetime
import os
from fastdtw import fastdtw
import scipy.spatial.distance as distance
from scipy.signal import spectrogram
import time
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

first = 0  # Змінна для першого запуску функції

def calculate_amplitude(audio_data):
    return np.max(np.abs(audio_data)) * 1000  # Розрахунок амплітуди аудіо

previous_time_below_threshold = None  # Попередній час нижче порогу

def print_amplitude(indata, frames, time_info, status):
    global amplitude_threshold, recording_started, recorded_frames, previous_time_below_threshold, first
    if first == 0:
        calculate_threshold(indata, frames, time, status);  # Розрахунок порогу при першому запуску
        first += 1
    amplitude = calculate_amplitude(indata)
    amplitude_text.delete(1.0, tk.END)
    amplitude_text.insert(tk.END, f"Amplitude: {amplitude:.2f}\n")
    amplitude_text.see(tk.END)

    log_amplitude_change(amplitude)

    if recording_started:
        recorded_frames.append(indata.copy())

    if amplitude > amplitude_threshold + 40 and not recording_started:
        start_recording(indata)  # Початок запису при перевищенні амплітуди
        previous_time_below_threshold = None  # Скидання лічильника тривалості
    elif amplitude > amplitude_threshold + 40 and recording_started:
        previous_time_below_threshold = None
    elif amplitude <= 10 and recording_started:
        if previous_time_below_threshold is None:
            previous_time_below_threshold = time.time()
        else:
            duration_below_threshold = time.time() - previous_time_below_threshold
            if duration_below_threshold > 0.1:  # Перевірка чи тривалість більше 1 секунди
                stop_recording()
                previous_time_below_threshold = None  # Скидання лічильника тривалості

def calculate_threshold(indata, frames, time_info, status):
    global amplitude_threshold, stream
    amplitude_threshold = 0
    start_time = time.time()
    amplitude_sum = 0
    samples_count = 0

    while time.time() - start_time < 5:
        amplitude = calculate_amplitude(indata)
        amplitude_sum += amplitude
        samples_count += 1

    if samples_count > 0:
        average_amplitude = amplitude_sum / samples_count
        amplitude_threshold = average_amplitude
        print(f"Amplitude Threshold set to: {amplitude_threshold}")

def log_amplitude_change(amplitude):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - Amplitude: {amplitude:.2f}\n"

    with open("logs.txt", "a") as log_file:
        log_file.write(log_entry)


def start_recording(indata):
    global recording_started, recorded_frames, output_counter, start_time
    recording_started = True
    start_time = datetime.now()

    print("Recording Started")
    recorded_frames = [indata.copy()]
    update_total_samples_label()


def stop_recording():
    global recording_started, recorded_frames, fs, output_counter, start_time, total_duration
    if recording_started:
        recording_started = False
        print("Recording Stopped")
        if recorded_frames:
            duration = (datetime.now() - start_time).total_seconds()
            if duration >= 0.4:
                output_filename = f"sample{output_counter}.wav"
                output_path = os.path.join("Sample", output_filename)
                frames = np.concatenate(recorded_frames, axis=0)
                wav.write(output_path, fs, frames)
                total_duration += duration
                output_counter += 1  # Інкремент лічильника лише при збереженні файлу
                update_total_samples_label()
                update_average_duration_label()
            else:
                print(f"Ignored recording with duration {duration} seconds")


def start_listening():
    global stream
    selected_device_index = devices_combobox.current()
    if selected_device_index:
        selected_device_info = devices[selected_device_index]
        selected_device_id = selected_device_info['index']
        stream = sd.InputStream(callback=print_amplitude, device=selected_device_id, channels=1, samplerate=fs)
        stream.start()
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
    else:
        print("Будь ласка, оберіть пристрій.")


def stop_listening():
    global stream
    if stream:
        stop_recording()
        stream.stop()
        stream.close()
        create_output_folder_and_modify_samples()
        coss_sim_values = compare_test_audio_with_samples()

        if coss_sim_values:
            most_similar_index = np.argmax(coss_sim_values)  # Знаходження індексу найбільшої відповідності
            most_similar_sample = f"output{most_similar_index + 1}.wav"  # Індекс починається з 0

            print(f"Найбільш схожий зразок на 'Test.wav' - {most_similar_sample}")
            print("Значення косинусної подібності:", coss_sim_values)
        else:
            print("Порівняння не можливе.")

    start_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)


def update_average_duration_label():
    global output_counter, total_duration
    average_duration = total_duration / output_counter if output_counter > 0 else 0
    average_duration_text.delete(1.0, tk.END)
    average_duration_text.insert(tk.END, f"Середня тривалість: {average_duration:.2f} секунд")


def update_total_samples_label():
    global output_counter
    total_samples_text.delete(1.0, tk.END)
    total_samples_text.insert(tk.END, f"Всього зразків: {output_counter}")


def create_output_folder_and_modify_samples():
    global total_duration, output_counter

    # Перевірка, чи існує тека 'Sample'
    sample_folder = "Sample"
    if not os.path.exists(sample_folder):
        print(f"'{sample_folder}' теця не існує.")
        return

    # Створення теки "output", якщо вона не існує
    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Пошук довжини найдовшого збереженого зразка
    max_sample_length = 0
    for i in range(1, output_counter + 1):
        sample_filename = f"sample{i}.wav"
        sample_path = os.path.join(sample_folder, sample_filename)
        if os.path.exists(sample_path):  # Перевірка, чи файл існує
            _, sample_data = wav.read(sample_path)
            max_sample_length = max(max_sample_length, len(sample_data))

    # Зміна та збереження всіх зразків так, щоб вони відповідали найдовшому зразку
    for i in range(1, output_counter + 1):
        sample_filename = f"sample{i}.wav"
        sample_path = os.path.join(sample_folder, sample_filename)

        if os.path.exists(sample_path):  # Перевірка, чи файл існує
            _, sample_data = wav.read(sample_path)

            # Підгонка довжини зразка до найдовшого зразка
            if len(sample_data) < max_sample_length:
                padding = np.zeros(max_sample_length - len(sample_data))
                sample_data = np.concatenate((sample_data, padding))

            # Збереження зміненого зразка у теку "output"
            output_filename = f"output{i}.wav"
            if i + 1 == output_counter:
                output_filename = f"Test.wav"
            output_path = os.path.join(output_folder, output_filename)
            wav.write(output_path, fs, sample_data)


def plot_spectrogram(audio_data, title):
    freq, time, spec = spectrogram(audio_data, fs)
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(time, freq, 10 * np.log10(spec), shading='gouraud')
    plt.title(title)
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Час [сек]')
    plt.colorbar(label='Інтенсивність [дБ]')
    plt.tight_layout()
    plt.show()

# Змінити функцію compare_test_audio_with_samples

def compare_test_audio_with_samples():
    # Завантаження файлу 'Test.wav' для порівняння
    test_audio_path = os.path.join("output", "Test.wav")
    if not os.path.exists(test_audio_path):
        print("Аудіофайл 'Test.wav' не знайдено.")
        return None

    _, test_audio_data = wav.read(test_audio_path)

    # Визначення змінних для зберігання балів схожості та спектрограм
    similarity_scores = []
    test_spec_freq, test_spec_time, test_spec = spectrogram(test_audio_data, fs)

    num_samples = output_counter - 1  # За винятком 'Test.wav'
    num_cols = 3  # Кількість стовпців в сітці
    num_rows = (num_samples + num_cols - 1) // num_cols  # Обчислення кількості необхідних рядків

    plt.figure(figsize=(10, 6))

    # Спектрограма для аудіофайлу Test.wav
    plt.subplot(num_rows, num_cols, 1)
    plt.specgram(test_audio_data, Fs=fs)
    plt.title('Спектрограма Test.wav')
    plt.xlabel('Час')
    plt.ylabel('Частота')

    # Порівняння аудіофайлу 'Test.wav' з кожним файлом зразка в теки 'output'
    for i in range(1, output_counter):  # Припускається, що файли названі як output1.wav, output2.wav, і т.д.
        sample_filename = f"output{i}.wav"
        sample_path = os.path.join("output", sample_filename)

        if os.path.exists(sample_path):
            _, sample_data = wav.read(sample_path)

            # Перевірка відповідності довжин аудіофайлу 'Test.wav' та зразка
            min_length = min(len(test_audio_data), len(sample_data))
            test_audio_data = test_audio_data[:min_length]
            sample_data = sample_data[:min_length]

            # Обчислення спектрограм для аудіофайлу 'Test.wav' та зразка
            sample_spec_freq, sample_spec_time, sample_spec = spectrogram(sample_data, fs)

            test_flat = test_spec.ravel().reshape(1, -1)  # Перетворення до 2D-масиву для cosine_similarity в sklearn
            sample_flat = sample_spec.ravel().reshape(1, -1)

            # Обчислення косинусної схожості
            cosine_sim = cosine_similarity(test_flat, sample_flat)[0][0]

            similarity_scores.append(cosine_sim)

            # Спектрограма для кожного зразка
            plt.subplot(num_rows, num_cols, i + 1)
            plt.specgram(sample_data, Fs=fs)
            plt.title(f'Спектрограма Зразка {i}')
            plt.xlabel('Час')
            plt.ylabel('Частота')

    plt.tight_layout()
    plt.show()

    return similarity_scores


# Налаштування інтерфейсу користувача
root = tk.Tk()
root.title("Амплітуда мікрофона")

# Випадаючий список пристроїв
devices = sd.query_devices()
devices_names = [f"{device['name']}" for device in devices]
devices_combobox = ttk.Combobox(root, values=devices_names, state="readonly", font=("Helvetica", 12))
default_device_index = [i for i, device in enumerate(devices) if
                        "Microphone" in device['name'] and device['max_input_channels'] > 0]
devices_combobox.current(default_device_index[0] if default_device_index else 0)
devices_combobox.pack(pady=10)

# Мітка та текст для середньої тривалості зразка
average_duration_label = tk.Label(root, text="Середня тривалість зразка:", font=("Helvetica", 12))
average_duration_label.pack()

average_duration_text = tk.Text(root, height=1, width=20, font=("Helvetica", 12))
average_duration_text.pack(pady=5, anchor='center', fill='x')

# Мітка та текст для загальної кількості збережених зразків
total_samples_label = tk.Label(root, text="Всього зразків збережено:", font=("Helvetica", 12))
total_samples_label.pack()

total_samples_text = tk.Text(root, height=1, width=20, font=("Helvetica", 12))
total_samples_text.pack(pady=5, anchor='center', fill='x')

# Кнопки
start_button = tk.Button(root, text="Почати прослуховування", command=start_listening, font=("Helvetica", 12))
start_button.pack(pady=5)

stop_button = tk.Button(root, text="Зупинити прослуховування", command=stop_listening, state=tk.DISABLED, font=("Helvetica", 12))
stop_button.pack(pady=5)

# Відображення амплітуди
amplitude_label = tk.Label(root, text="Амплітуда:", font=("Helvetica", 12))
amplitude_label.pack()

amplitude_text = tk.Text(root, height=2, width=20, font=("Helvetica", 12))
amplitude_text.pack(pady=5, anchor='center', fill='x')

# Глобальні змінні
recording_started = False
recorded_frames = None
fs = 44100
amplitude_threshold = 40
stream = None
output_counter = 1
start_time = None
total_duration = 0

# Створення папки "Sample", якщо вона не існує
if not os.path.exists("Sample"):
    os.makedirs("Sample")

root.mainloop()