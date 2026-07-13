// Live Parakeet dictation from the default PipeWire source.
//
// This is intentionally Linux/PipeWire-only. Native unsandboxed command-line
// applications generally connect to the user's PipeWire session directly; the
// desktop portal path is mostly for sandboxed apps.

#include "ggml-backend.h"
#include "parakeet.h"

#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/param/audio/raw.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cerrno>
#include <memory>

#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

struct lyssna_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t gpu_device = 0;

    int32_t block_ms      = 250;
    int32_t pre_speech_ms = 300;
    int32_t silence_ms    = 800;
    int32_t min_speech_ms = 400;
    int32_t max_speech_ms = 6000;

    float vad_thold = 0.005f;

    bool use_gpu      = true;
    bool manual       = false;
    bool print_events = true;
    bool print_energy = false;
    bool print_timings = false;

    bool daemon       = false;
    bool type_output  = true;

    int32_t model_idle_timeout_s = 300;

    std::string model  = "models/ggml-parakeet-tdt-0.6b-v3-q4_k.bin";
    std::string target = "";
    std::string socket = "";
    std::string typer  = "wtype";
};

static std::atomic_bool g_running(true);
static std::atomic_int  g_manual_events(0);

static void signal_handler(int) {
    g_running = false;
}

static void manual_input_thread() {
    std::string line;
    while (g_running && std::getline(std::cin, line)) {
        g_manual_events.fetch_add(1);
    }
}

static char * requires_value_error(const std::string & arg) {
    fprintf(stderr, "error: argument %s requires value\n", arg.c_str());
    exit(1);
}

static std::string default_socket_path() {
    const char * runtime_dir = std::getenv("XDG_RUNTIME_DIR");
    if (!runtime_dir || !runtime_dir[0]) {
        fprintf(stderr, "error: XDG_RUNTIME_DIR is not set; use --socket FILE\n");
        exit(1);
    }
    return std::string(runtime_dir) + "/lyssna.sock";
}

static void print_usage(char ** argv, const lyssna_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help              show this help message and exit\n");
    fprintf(stderr, "  -m FILE,  --model FILE        [%-7s] model path\n", params.model.c_str());
    fprintf(stderr, "  -t N,     --threads N         [%-7d] number of threads\n", params.n_threads);
    fprintf(stderr, "  -ng,      --no-gpu            [%-7s] disable GPU\n", params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -dev N,   --device N          [%-7d] GPU device\n", params.gpu_device);
    fprintf(stderr, "            --target NAME       [%-7s] PipeWire target object\n", params.target.c_str());
    fprintf(stderr, "            --block-ms N        [%-7d] capture/VAD block size\n", params.block_ms);
    fprintf(stderr, "            --pre-speech-ms N   [%-7d] audio kept before speech starts\n", params.pre_speech_ms);
    fprintf(stderr, "            --silence-ms N      [%-7d] silence that ends an utterance\n", params.silence_ms);
    fprintf(stderr, "            --min-speech-ms N   [%-7d] minimum utterance length\n", params.min_speech_ms);
    fprintf(stderr, "            --max-speech-ms N   [%-7d] force decode after this much speech\n", params.max_speech_ms);
    fprintf(stderr, "            --vad-thold N       [%-7.4f] RMS threshold for speech\n", params.vad_thold);
    fprintf(stderr, "            --manual            [%-7s] press Enter to start/stop recording\n", params.manual ? "true" : "false");
    fprintf(stderr, "            --no-events         [%-7s] do not print speech/decode events\n", params.print_events ? "false" : "true");
    fprintf(stderr, "            --print-energy      [%-7s] print per-block RMS values\n", params.print_energy ? "true" : "false");
    fprintf(stderr, "            --print-timings     [%-7s] print Parakeet timings after each utterance\n", params.print_timings ? "true" : "false");
    fprintf(stderr, "            --daemon            [%-7s] run the push-to-talk control service\n", params.daemon ? "true" : "false");
    fprintf(stderr, "            --socket FILE       [auto   ] control socket path\n");
    fprintf(stderr, "            --model-idle N      [%-7d] unload model after N idle seconds (-1 keeps it loaded)\n", params.model_idle_timeout_s);
    fprintf(stderr, "            --typer FILE        [%-7s] Wayland typing program\n", params.typer.c_str());
    fprintf(stderr, "            --no-type           [%-7s] print results without typing them\n", params.type_output ? "false" : "true");
    fprintf(stderr, "\ncontrol commands:\n");
    fprintf(stderr, "  %s [--socket FILE] start|stop|cancel|status|unload\n", argv[0]);
    fprintf(stderr, "\n");
}

static bool parse_params(int argc, char ** argv, lyssna_params & params) {
    if (const char * env_device = std::getenv("LYSSNA_DEVICE")) {
        params.gpu_device = std::stoi(env_device);
    }

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv, params);
            exit(0);
        }

#define ARGV_NEXT (((i + 1) < argc) ? argv[++i] : requires_value_error(arg))
        if (arg == "-m" || arg == "--model") {
            params.model = ARGV_NEXT;
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(ARGV_NEXT);
        } else if (arg == "-ng" || arg == "--no-gpu") {
            params.use_gpu = false;
        } else if (arg == "-dev" || arg == "--device") {
            params.gpu_device = std::stoi(ARGV_NEXT);
        } else if (arg == "--target") {
            params.target = ARGV_NEXT;
        } else if (arg == "--block-ms") {
            params.block_ms = std::stoi(ARGV_NEXT);
        } else if (arg == "--pre-speech-ms") {
            params.pre_speech_ms = std::stoi(ARGV_NEXT);
        } else if (arg == "--silence-ms") {
            params.silence_ms = std::stoi(ARGV_NEXT);
        } else if (arg == "--min-speech-ms") {
            params.min_speech_ms = std::stoi(ARGV_NEXT);
        } else if (arg == "--max-speech-ms") {
            params.max_speech_ms = std::stoi(ARGV_NEXT);
        } else if (arg == "--vad-thold") {
            params.vad_thold = std::stof(ARGV_NEXT);
        } else if (arg == "--manual") {
            params.manual = true;
        } else if (arg == "--no-events") {
            params.print_events = false;
        } else if (arg == "--print-energy") {
            params.print_energy = true;
        } else if (arg == "--print-timings") {
            params.print_timings = true;
        } else if (arg == "--daemon") {
            params.daemon = true;
        } else if (arg == "--socket") {
            params.socket = ARGV_NEXT;
        } else if (arg == "--model-idle") {
            params.model_idle_timeout_s = std::stoi(ARGV_NEXT);
        } else if (arg == "--typer") {
            params.typer = ARGV_NEXT;
        } else if (arg == "--no-type") {
            params.type_output = false;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            print_usage(argv, params);
            return false;
        }
    }

    params.block_ms      = std::max(50, params.block_ms);
    params.pre_speech_ms = std::max(0, params.pre_speech_ms);
    params.silence_ms    = std::max(params.block_ms, params.silence_ms);
    params.min_speech_ms = std::max(params.block_ms, params.min_speech_ms);
    params.max_speech_ms = std::max(params.min_speech_ms, params.max_speech_ms);
    params.vad_thold     = std::max(0.0f, params.vad_thold);

    return true;
}

class pipewire_capture {
public:
    pipewire_capture(int sample_rate, int max_queue_ms, const std::string & target) :
        m_sample_rate(sample_rate),
        m_max_queue_samples((sample_rate * max_queue_ms) / 1000),
        m_target(target) {
    }

    bool init() {
        pw_init(nullptr, nullptr);

        m_loop = pw_thread_loop_new("lyssna", nullptr);
        if (!m_loop) {
            fprintf(stderr, "%s: failed to create PipeWire thread loop\n", __func__);
            return false;
        }

        m_context = pw_context_new(pw_thread_loop_get_loop(m_loop), nullptr, 0);
        if (!m_context) {
            fprintf(stderr, "%s: failed to create PipeWire context\n", __func__);
            return false;
        }

        m_core = pw_context_connect(m_context, nullptr, 0);
        if (!m_core) {
            fprintf(stderr, "%s: failed to connect to PipeWire\n", __func__);
            return false;
        }

        pw_properties * props = pw_properties_new(
                PW_KEY_MEDIA_TYPE,     "Audio",
                PW_KEY_MEDIA_CATEGORY, "Capture",
                PW_KEY_MEDIA_ROLE,     "Speech",
                PW_KEY_APP_NAME,       "lyssna",
                nullptr);
        if (!m_target.empty()) {
            pw_properties_set(props, PW_KEY_TARGET_OBJECT, m_target.c_str());
        }

        m_stream = pw_stream_new(m_core, "lyssna", props);
        if (!m_stream) {
            fprintf(stderr, "%s: failed to create PipeWire stream\n", __func__);
            return false;
        }

        pw_stream_add_listener(m_stream, &m_stream_listener, &stream_events, this);

        uint8_t buffer[1024];
        spa_pod_builder builder = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

        spa_audio_info_raw audio_info = {};
        audio_info.format = SPA_AUDIO_FORMAT_F32;
        audio_info.rate = m_sample_rate;
        audio_info.channels = 1;
        audio_info.position[0] = SPA_AUDIO_CHANNEL_MONO;

        const spa_pod * stream_params[1];
        stream_params[0] = spa_format_audio_raw_build(&builder, SPA_PARAM_EnumFormat, &audio_info);

        const int ret = pw_stream_connect(
                m_stream,
                PW_DIRECTION_INPUT,
                PW_ID_ANY,
                (pw_stream_flags) (PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS),
                stream_params,
                1);
        if (ret < 0) {
            fprintf(stderr, "%s: failed to connect PipeWire stream: %s\n", __func__, std::strerror(-ret));
            return false;
        }

        if (pw_thread_loop_start(m_loop) < 0) {
            fprintf(stderr, "%s: failed to start PipeWire thread loop\n", __func__);
            return false;
        }

        m_started = true;
        return true;
    }

    ~pipewire_capture() {
        if (m_started && m_loop) {
            pw_thread_loop_stop(m_loop);
        }
        if (m_stream) {
            spa_hook_remove(&m_stream_listener);
            pw_stream_destroy(m_stream);
        }
        if (m_core) {
            pw_core_disconnect(m_core);
        }
        if (m_context) {
            pw_context_destroy(m_context);
        }
        if (m_loop) {
            pw_thread_loop_destroy(m_loop);
        }
        pw_deinit();
    }

    void pop_wait(int n_samples, std::vector<float> & out) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond.wait(lock, [&] {
            return !g_running || (int) m_samples.size() >= n_samples;
        });

        const int n = std::min<int>(n_samples, m_samples.size());
        out.resize(n);
        for (int i = 0; i < n; ++i) {
            out[i] = m_samples.front();
            m_samples.pop_front();
        }
    }

    std::vector<float> drain() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<float> result;
        result.reserve(m_samples.size());
        while (!m_samples.empty()) {
            result.push_back(m_samples.front());
            m_samples.pop_front();
        }
        return result;
    }

private:
    static void on_process(void * user_data) {
        pipewire_capture * self = (pipewire_capture *) user_data;
        pw_buffer * buffer = nullptr;

        while ((buffer = pw_stream_dequeue_buffer(self->m_stream)) != nullptr) {
            spa_buffer * spa_buffer = buffer->buffer;
            if (spa_buffer && spa_buffer->n_datas > 0 && spa_buffer->datas[0].data) {
                spa_data & data = spa_buffer->datas[0];
                const uint32_t offset = data.chunk ? data.chunk->offset : 0;
                const uint32_t size   = data.chunk ? data.chunk->size   : data.maxsize;

                const float * src = (const float *) ((const uint8_t *) data.data + offset);
                const int n_samples = size / sizeof(float);

                {
                    std::lock_guard<std::mutex> lock(self->m_mutex);
                    for (int i = 0; i < n_samples; ++i) {
                        self->m_samples.push_back(src[i]);
                    }

                    while (self->m_samples.size() > self->m_max_queue_samples) {
                        self->m_samples.pop_front();
                    }
                }

                self->m_cond.notify_one();
            }

            pw_stream_queue_buffer(self->m_stream, buffer);
        }
    }

    static void on_state_changed(void *, pw_stream_state, pw_stream_state state, const char * error) {
        if (state == PW_STREAM_STATE_ERROR) {
            fprintf(stderr, "PipeWire stream error: %s\n", error ? error : "unknown");
            g_running = false;
        }
    }

    static const pw_stream_events stream_events;

    int m_sample_rate = 0;
    size_t m_max_queue_samples = 0;
    std::string m_target;

    pw_thread_loop * m_loop = nullptr;
    pw_context * m_context = nullptr;
    pw_core * m_core = nullptr;
    pw_stream * m_stream = nullptr;
    spa_hook m_stream_listener = {};
    bool m_started = false;

    std::mutex m_mutex;
    std::condition_variable m_cond;
    std::deque<float> m_samples;
};

const pw_stream_events pipewire_capture::stream_events = {
    PW_VERSION_STREAM_EVENTS,
    nullptr,
    pipewire_capture::on_state_changed,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    pipewire_capture::on_process,
    nullptr,
    nullptr,
    nullptr,
};

static float rms_energy(const std::vector<float> & pcm) {
    if (pcm.empty()) {
        return 0.0f;
    }

    double sum = 0.0;
    for (float s : pcm) {
        sum += double(s) * double(s);
    }

    return std::sqrt(sum / pcm.size());
}

static void append_limited(std::vector<float> & dst, const std::vector<float> & src, int max_samples) {
    dst.insert(dst.end(), src.begin(), src.end());
    if ((int) dst.size() > max_samples) {
        dst.erase(dst.begin(), dst.end() - max_samples);
    }
}

static std::string trim_text(const char * text) {
    if (!text) {
        return "";
    }

    std::string result(text);
    while (!result.empty() && std::isspace((unsigned char) result.front())) {
        result.erase(result.begin());
    }
    while (!result.empty() && std::isspace((unsigned char) result.back())) {
        result.pop_back();
    }
    return result;
}

static bool transcribe_utterance(
        parakeet_context * ctx,
        const lyssna_params & params,
        const std::vector<float> & utterance,
        const char * reason) {
    if (utterance.empty()) {
        return true;
    }

    const double utterance_s = utterance.size() / (double) PARAKEET_SAMPLE_RATE;
    if (params.print_events) {
        fprintf(stderr, "transcribing %.2fs utterance (%s)\n", utterance_s, reason);
    }

    if (params.print_timings) {
        parakeet_reset_timings(ctx);
    }

    const auto t0 = std::chrono::steady_clock::now();

    parakeet_full_params full_params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
    full_params.n_threads = params.n_threads;
    full_params.no_context = true;

    const int ret = parakeet_full(ctx, full_params, utterance.data(), utterance.size());
    if (ret != 0) {
        fprintf(stderr, "error: failed to transcribe utterance, ret = %d\n", ret);
        return false;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_s = std::chrono::duration<double>(t1 - t0).count();

    const int n_segments = parakeet_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const std::string text = trim_text(parakeet_full_get_segment_text(ctx, i));
        if (!text.empty()) {
            printf("%s\n", text.c_str());
            fflush(stdout);
        }
    }

    if (params.print_events) {
        fprintf(stderr, "transcribed in %.2fs\n", elapsed_s);
    }

    if (params.print_timings) {
        parakeet_print_timings(ctx);
    }

    return true;
}

static bool write_all(int fd, const std::string & data) {
    size_t offset = 0;
    while (offset < data.size()) {
        const ssize_t n = write(fd, data.data() + offset, data.size() - offset);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        offset += n;
    }
    return true;
}

static std::string read_line(int fd) {
    std::string result;
    char c;
    while (result.size() < 4096) {
        const ssize_t n = read(fd, &c, 1);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0 || c == '\n') {
            break;
        }
        result.push_back(c);
    }
    return result;
}

static int connect_control_socket(const std::string & socket_path) {
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        fprintf(stderr, "error: socket: %s\n", std::strerror(errno));
        return -1;
    }

    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        fprintf(stderr, "error: socket path is too long: %s\n", socket_path.c_str());
        close(fd);
        return -1;
    }
    std::memcpy(addr.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (connect(fd, (sockaddr *) &addr, sizeof(addr)) < 0) {
        fprintf(stderr, "error: cannot connect to %s: %s\n", socket_path.c_str(), std::strerror(errno));
        close(fd);
        return -1;
    }
    return fd;
}

static bool is_control_command(const std::string & arg) {
    return arg == "start" || arg == "stop" || arg == "cancel" || arg == "status" || arg == "unload";
}

static int run_control_command(int argc, char ** argv) {
    std::string command;
    std::string socket_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--socket" && i + 1 < argc) {
            socket_path = argv[++i];
        } else if (is_control_command(arg)) {
            command = arg;
        }
    }
    if (command.empty()) {
        return -1;
    }
    if (socket_path.empty()) {
        socket_path = default_socket_path();
    }

    const int fd = connect_control_socket(socket_path);
    if (fd < 0) {
        return 1;
    }
    if (!write_all(fd, command + "\n")) {
        fprintf(stderr, "error: failed to send command: %s\n", std::strerror(errno));
        close(fd);
        return 1;
    }
    const std::string response = read_line(fd);
    close(fd);
    if (!response.empty()) {
        printf("%s\n", response.c_str());
    }
    return response.rfind("OK", 0) == 0 ? 0 : 1;
}

static int create_control_socket(const std::string & socket_path) {
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        fprintf(stderr, "error: socket: %s\n", std::strerror(errno));
        return -1;
    }

    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path)) {
        fprintf(stderr, "error: socket path is too long: %s\n", socket_path.c_str());
        close(fd);
        return -1;
    }
    std::memcpy(addr.sun_path, socket_path.c_str(), socket_path.size() + 1);
    unlink(socket_path.c_str());
    if (bind(fd, (sockaddr *) &addr, sizeof(addr)) < 0 || chmod(socket_path.c_str(), 0600) < 0 || listen(fd, 8) < 0) {
        fprintf(stderr, "error: cannot create %s: %s\n", socket_path.c_str(), std::strerror(errno));
        close(fd);
        unlink(socket_path.c_str());
        return -1;
    }
    return fd;
}

static std::string transcribe_to_text(
        parakeet_context * ctx,
        const lyssna_params & params,
        const std::vector<float> & utterance) {
    parakeet_full_params full_params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
    full_params.n_threads = params.n_threads;
    full_params.no_context = true;
    if (parakeet_full(ctx, full_params, utterance.data(), utterance.size()) != 0) {
        return {};
    }

    std::string result;
    const int n_segments = parakeet_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const std::string text = trim_text(parakeet_full_get_segment_text(ctx, i));
        if (!text.empty()) {
            if (!result.empty()) {
                result.push_back(' ');
            }
            result += text;
        }
    }
    return result;
}

static bool type_text(const lyssna_params & params, const std::string & text) {
    if (!params.type_output) {
        printf("%s\n", text.c_str());
        fflush(stdout);
        return true;
    }

    const std::string output = text + " ";
    const pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "error: fork: %s\n", std::strerror(errno));
        return false;
    }
    if (pid == 0) {
        execlp(params.typer.c_str(), params.typer.c_str(), output.c_str(), (char *) nullptr);
        fprintf(stderr, "error: cannot execute %s: %s\n", params.typer.c_str(), std::strerror(errno));
        _exit(127);
    }

    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fprintf(stderr, "error: %s failed\n", params.typer.c_str());
        return false;
    }
    return true;
}

static int run_daemon(lyssna_params params) {
    if (params.socket.empty()) {
        params.socket = default_socket_path();
    }
    const int server_fd = create_control_socket(params.socket);
    if (server_fd < 0) {
        return 1;
    }

    fprintf(stderr, "Dictation control socket: %s\n", params.socket.c_str());
    fprintf(stderr, "Model will load on first transcription and unload after %d idle seconds.\n", params.model_idle_timeout_s);

    std::unique_ptr<pipewire_capture> capture;
    parakeet_context * ctx = nullptr;
    std::chrono::steady_clock::time_point recording_started;
    std::chrono::steady_clock::time_point model_last_used;
    bool recording = false;

    auto unload_model = [&]() {
        if (ctx) {
            fprintf(stderr, "Unloading Parakeet model.\n");
            parakeet_free(ctx);
            ctx = nullptr;
        }
    };

    auto finish_recording = [&](const char * reason) {
        if (!recording || !capture) {
            return;
        }
        std::vector<float> utterance = capture->drain();
        capture.reset();
        recording = false;

        const int min_samples = (PARAKEET_SAMPLE_RATE * params.min_speech_ms) / 1000;
        if ((int) utterance.size() < min_samples) {
            fprintf(stderr, "Discarding short recording (%.2fs).\n", utterance.size() / (double) PARAKEET_SAMPLE_RATE);
            return;
        }

        fprintf(stderr, "Decoding %.2fs recording (%s).\n", utterance.size() / (double) PARAKEET_SAMPLE_RATE, reason);
        if (!ctx) {
            ggml_backend_load_all();
            parakeet_context_params ctx_params = parakeet_context_default_params();
            ctx_params.use_gpu = params.use_gpu;
            ctx_params.gpu_device = params.gpu_device;
            fprintf(stderr, "Loading Parakeet model from: %s\n", params.model.c_str());
            ctx = parakeet_init_from_file_with_params(params.model.c_str(), ctx_params);
            if (!ctx) {
                fprintf(stderr, "error: failed to load model\n");
                return;
            }
        }

        const std::string text = transcribe_to_text(ctx, params, utterance);
        model_last_used = std::chrono::steady_clock::now();
        if (text.empty()) {
            fprintf(stderr, "No text recognized.\n");
            return;
        }
        fprintf(stderr, "Recognized: %s\n", text.c_str());
        type_text(params, text);
    };

    while (g_running) {
        pollfd pfd = { server_fd, POLLIN, 0 };
        const int poll_result = poll(&pfd, 1, 100);
        if (poll_result < 0 && errno != EINTR) {
            fprintf(stderr, "error: poll: %s\n", std::strerror(errno));
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        if (recording && now - recording_started >= std::chrono::milliseconds(params.max_speech_ms)) {
            finish_recording("maximum duration");
        }
        if (ctx && !recording && params.model_idle_timeout_s >= 0 &&
                now - model_last_used >= std::chrono::seconds(params.model_idle_timeout_s)) {
            unload_model();
        }

        if (poll_result <= 0 || !(pfd.revents & POLLIN)) {
            continue;
        }

        const int client_fd = accept4(server_fd, nullptr, nullptr, SOCK_CLOEXEC);
        if (client_fd < 0) {
            if (errno != EINTR) {
                fprintf(stderr, "error: accept: %s\n", std::strerror(errno));
            }
            continue;
        }
        const std::string command = read_line(client_fd);

        if (command == "start") {
            if (recording) {
                write_all(client_fd, "ERR already recording\n");
            } else {
                capture.reset(new pipewire_capture(PARAKEET_SAMPLE_RATE, params.max_speech_ms + 2000, params.target));
                if (!capture->init()) {
                    capture.reset();
                    write_all(client_fd, "ERR PipeWire capture failed\n");
                } else {
                    recording = true;
                    recording_started = std::chrono::steady_clock::now();
                    fprintf(stderr, "Recording started.\n");
                    write_all(client_fd, "OK recording\n");
                }
            }
        } else if (command == "stop") {
            if (!recording) {
                write_all(client_fd, "ERR not recording\n");
            } else {
                write_all(client_fd, "OK decoding\n");
                close(client_fd);
                finish_recording("key release");
                continue;
            }
        } else if (command == "cancel") {
            capture.reset();
            recording = false;
            fprintf(stderr, "Recording cancelled.\n");
            write_all(client_fd, "OK idle\n");
        } else if (command == "status") {
            write_all(client_fd, std::string("OK ") + (recording ? "recording" : "idle") + (ctx ? " model-loaded\n" : " model-unloaded\n"));
        } else if (command == "unload") {
            if (recording) {
                write_all(client_fd, "ERR recording\n");
            } else {
                unload_model();
                write_all(client_fd, "OK model-unloaded\n");
            }
        } else {
            write_all(client_fd, "ERR unknown command\n");
        }
        close(client_fd);
    }

    capture.reset();
    unload_model();
    close(server_fd);
    unlink(params.socket.c_str());
    return 0;
}

int main(int argc, char ** argv) {
    const int control_result = run_control_command(argc, argv);
    if (control_result >= 0) {
        return control_result;
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    lyssna_params params;
    if (!parse_params(argc, argv, params)) {
        return 1;
    }

    if (params.daemon) {
        return run_daemon(params);
    }

    ggml_backend_load_all();

    parakeet_context_params ctx_params = parakeet_context_default_params();
    ctx_params.use_gpu    = params.use_gpu;
    ctx_params.gpu_device = params.gpu_device;

    fprintf(stderr, "Loading Parakeet model from: %s\n", params.model.c_str());
    parakeet_context * ctx = parakeet_init_from_file_with_params(params.model.c_str(), ctx_params);
    if (!ctx) {
        fprintf(stderr, "error: failed to load model '%s'\n", params.model.c_str());
        return 1;
    }

    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
            params.n_threads,
            (int32_t) std::thread::hardware_concurrency(),
            parakeet_print_system_info());

    const int block_samples      = (PARAKEET_SAMPLE_RATE * params.block_ms) / 1000;
    const int pre_speech_samples = (PARAKEET_SAMPLE_RATE * params.pre_speech_ms) / 1000;
    const int silence_samples    = (PARAKEET_SAMPLE_RATE * params.silence_ms) / 1000;
    const int min_speech_samples = (PARAKEET_SAMPLE_RATE * params.min_speech_ms) / 1000;
    const int max_speech_samples = (PARAKEET_SAMPLE_RATE * params.max_speech_ms) / 1000;

    pipewire_capture capture(PARAKEET_SAMPLE_RATE, params.max_speech_ms + params.pre_speech_ms + 5000, params.target);
    if (!capture.init()) {
        parakeet_free(ctx);
        return 1;
    }

    fprintf(stderr, "Capturing from PipeWire default source at %d Hz. Press Ctrl-C to stop.\n", PARAKEET_SAMPLE_RATE);
    if (params.manual) {
        fprintf(stderr, "Manual mode: press Enter to start recording, press Enter again to stop and transcribe.\n");
        std::thread(manual_input_thread).detach();
    } else {
        fprintf(stderr, "VAD threshold %.4f, silence %d ms, max utterance %d ms.\n",
                params.vad_thold, params.silence_ms, params.max_speech_ms);
    }

    std::vector<float> block;
    std::vector<float> pre_speech;
    std::vector<float> utterance;

    bool in_speech = false;
    int silence_seen = 0;
    bool manual_recording = false;

    while (g_running) {
        capture.pop_wait(block_samples, block);
        if (block.empty()) {
            continue;
        }

        if (params.manual) {
            int events = g_manual_events.exchange(0);
            while (events-- > 0) {
                manual_recording = !manual_recording;
                if (manual_recording) {
                    utterance.clear();
                    if (params.print_events) {
                        fprintf(stderr, "recording started\n");
                    }
                } else {
                    if (params.print_events) {
                        fprintf(stderr, "recording stopped\n");
                    }
                    if ((int) utterance.size() >= min_speech_samples) {
                        if (!transcribe_utterance(ctx, params, utterance, "manual")) {
                            g_running = false;
                            break;
                        }
                    }
                    utterance.clear();
                }
            }

            if (!g_running) {
                break;
            }

            if (!manual_recording) {
                continue;
            }

            utterance.insert(utterance.end(), block.begin(), block.end());

            if ((int) utterance.size() >= max_speech_samples) {
                if (!transcribe_utterance(ctx, params, utterance, "max duration")) {
                    break;
                }
                utterance.clear();
                manual_recording = false;
                if (params.print_events) {
                    fprintf(stderr, "recording stopped at max duration; press Enter to start again\n");
                }
            }

            continue;
        }

        const float energy = rms_energy(block);
        const bool speech = energy >= params.vad_thold;

        if (params.print_energy) {
            fprintf(stderr, "rms=%0.6f speech=%d\n", energy, speech ? 1 : 0);
        }

        if (!in_speech) {
            if (speech) {
                in_speech = true;
                silence_seen = 0;
                utterance = pre_speech;
                pre_speech.clear();
                if (params.print_events) {
                    fprintf(stderr, "speech started (rms=%0.6f)\n", energy);
                }
            } else {
                append_limited(pre_speech, block, pre_speech_samples);
                continue;
            }
        }

        utterance.insert(utterance.end(), block.begin(), block.end());

        if (speech) {
            silence_seen = 0;
        } else {
            silence_seen += block.size();
        }

        const bool silence_done = silence_seen >= silence_samples && (int) utterance.size() >= min_speech_samples;
        const bool max_done     = (int) utterance.size() >= max_speech_samples;

        if (silence_done || max_done) {
            if ((int) utterance.size() >= min_speech_samples) {
                if (!transcribe_utterance(ctx, params, utterance, silence_done ? "silence" : "max duration")) {
                    break;
                }
            }

            utterance.clear();
            pre_speech.clear();
            in_speech = false;
            silence_seen = 0;
        }
    }

    if (!utterance.empty() && (int) utterance.size() >= min_speech_samples) {
        transcribe_utterance(ctx, params, utterance, "exit");
    }

    if (params.print_timings) {
        parakeet_print_timings(ctx);
    }
    parakeet_free(ctx);

    return 0;
}
