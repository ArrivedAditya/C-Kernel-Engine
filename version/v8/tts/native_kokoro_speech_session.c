/* One-request Kokoro speech supervisor. Model execution stays in the generated
 * native host; this process owns request replacement and optional playback. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

enum phase { IDLE, SYNTHESIZING, PLAYING };

typedef struct {
    enum phase phase;
    pid_t child;
    unsigned long long id;
    char wav[4096];
} request;

static int event(unsigned long long id, const char *kind, const char *detail) {
    if (detail) printf("EVENT\t%llu\t%s\t%s\n", id, kind, detail);
    else printf("EVENT\t%llu\t%s\n", id, kind);
    return fflush(stdout) == 0;
}

static void stop_child(request *active) {
    if (active->child <= 0) return;
    /* Each child owns a process group, including any player descendants. */
    if (kill(-active->child, SIGKILL) != 0 && errno != ESRCH)
        perror("Kokoro session child kill");
    while (waitpid(active->child, NULL, 0) < 0 && errno == EINTR) {}
    active->child = 0;
}

static void cancel(request *active) {
    if (active->phase == IDLE) return;
    stop_child(active);
    if (active->wav[0]) unlink(active->wav);
    event(active->id, "CANCELLED", NULL);
    active->phase = IDLE;
}

static pid_t launch(char *const args[]) {
    pid_t child = fork();
    if (child == 0) {
        if (setpgid(0, 0) != 0) _exit(127);
        int null_input = open("/dev/null", O_RDONLY);
        if (null_input < 0 || dup2(null_input, STDIN_FILENO) < 0 ||
            dup2(STDERR_FILENO, STDOUT_FILENO) < 0) _exit(127);
        if (null_input != STDIN_FILENO) close(null_input);
        execv(args[0], args);
        _exit(127);
    }
    if (child > 0 && setpgid(child, child) != 0 && errno != EACCES)
        perror("Kokoro session process group");
    return child;
}

static void reap(request *active, const char *player) {
    if (active->phase == IDLE) return;
    int status = 0;
    pid_t result = waitpid(active->child, &status, WNOHANG);
    if (result == 0) return;
    if (result < 0) {
        if (errno == EINTR) return;
        status = -1;
    }
    active->child = 0;
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        unlink(active->wav);
        event(active->id, "ERROR", "native child failed");
        active->phase = IDLE;
        return;
    }
    if (active->phase == PLAYING) {
        event(active->id, "COMPLETED", NULL);
        active->phase = IDLE;
        return;
    }
    struct stat info;
    if (stat(active->wav, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 44) {
        unlink(active->wav);
        event(active->id, "ERROR", "native WAV missing or empty");
        active->phase = IDLE;
        return;
    }
    if (!event(active->id, "WAV_READY", active->wav)) {
        active->phase = IDLE;
        return;
    }
    if (!player) {
        event(active->id, "COMPLETED", NULL);
        active->phase = IDLE;
        return;
    }
    char *const args[] = {(char *)player, active->wav, NULL};
    active->child = launch(args);
    if (active->child < 0) {
        event(active->id, "ERROR", "player launch failed");
        active->phase = IDLE;
    } else {
        active->phase = PLAYING;
        event(active->id, "PLAYING", NULL);
    }
}

static int seed_valid(const char *seed) {
    if (!*seed) return 0;
    unsigned long long value = 0;
    for (const unsigned char *p = (const unsigned char *)seed; *p; ++p) {
        if (*p < '0' || *p > '9' || value > (UINT32_MAX - (*p - '0')) / 10)
            return 0;
        value = value * 10 + (*p - '0');
    }
    return 1;
}

static void command(char *line, request *active, unsigned long long *next_id,
                    const char *frontend, const char *bundle, const char *lexicon,
                    const char *directory, int *running) {
    if (!strcmp(line, "STOP")) {
        cancel(active);
        return;
    }
    if (!strcmp(line, "QUIT")) {
        cancel(active);
        *running = 0;
        return;
    }
    if (strncmp(line, "SPEAK\t", 6)) {
        event(0, "ERROR", "expected SPEAK, STOP or QUIT");
        return;
    }
    char *seed = line + 6;
    char *text = strchr(seed, '\t');
    if (!text) {
        event(0, "ERROR", "missing text");
        return;
    }
    *text++ = 0;
    if (!seed_valid(seed) || !*text || strlen(text) > 512) {
        event(0, "ERROR", "invalid seed or text length");
        return;
    }
    cancel(active);
    if (*next_id == 0) {
        event(0, "ERROR", "request ID exhausted");
        return;
    }
    active->id = (*next_id)++;
    int length = snprintf(active->wav, sizeof(active->wav), "%s/%llu.wav",
                          directory, active->id);
    if (length <= 0 || (size_t)length >= sizeof(active->wav)) {
        event(active->id, "ERROR", "WAV path too long");
        active->phase = IDLE;
        return;
    }
    char *const args[] = {(char *)frontend, (char *)bundle, (char *)lexicon,
                          active->wav, seed, text, NULL};
    active->child = launch(args);
    if (active->child < 0) {
        event(active->id, "ERROR", "frontend launch failed");
        active->phase = IDLE;
        return;
    }
    active->phase = SYNTHESIZING;
    event(active->id, "STARTED", NULL);
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: %s native-kokoro-text bundle lexicon output-root [player|-]\n",
                argv[0]);
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);
    const char *player = argc == 6 && strcmp(argv[5], "-") ? argv[5] : NULL;
    char directory[4096];
    int length = snprintf(directory, sizeof(directory),
                          "%s/.cke-kokoro-session-XXXXXX", argv[4]);
    if (length <= 0 || (size_t)length >= sizeof(directory) || !mkdtemp(directory)) {
        perror("Kokoro session output directory");
        return 3;
    }
    printf("READY\t%s\n", directory);
    if (fflush(stdout) != 0) return 4;
    request active = {0};
    unsigned long long next_id = 1;
    char input[1024];
    size_t used = 0;
    int discarding = 0, running = 1;
    while (running && !ferror(stdout)) {
        reap(&active, player);
        struct pollfd fd = {.fd = STDIN_FILENO, .events = POLLIN | POLLHUP};
        int ready = poll(&fd, 1, active.phase == IDLE ? -1 : 20);
        if (ready < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (!ready) continue;
        char bytes[256];
        ssize_t count = read(STDIN_FILENO, bytes, sizeof(bytes));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        for (ssize_t i = 0; i < count && running; ++i) {
            if (bytes[i] == '\n') {
                if (discarding) event(0, "ERROR", "invalid or oversized command");
                else {
                    input[used] = 0;
                    command(input, &active, &next_id, argv[1], argv[2],
                            argv[3], directory, &running);
                }
                used = 0;
                discarding = 0;
            } else if (bytes[i] == '\0') {
                discarding = 1;
            } else if (!discarding && used < sizeof(input) - 1) {
                input[used++] = bytes[i];
            } else {
                discarding = 1;
            }
        }
    }
    cancel(&active);
    return ferror(stdout) ? 4 : 0;
}
