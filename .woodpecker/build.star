registry = "registry.session.codes/"

events = ["push", "pull_request", "tag", "deployment", "cron", "manual"]

apt_get = "apt-get -o=Dpkg::Use-Pty=0 -q"

default_deps = ["g++", "pkg-config", "libsodium-dev", "libfmt-dev"]

default_cmake = {
    "CMAKE_BUILD_TYPE": "Release",
    "CMAKE_COLOR_DIAGNOSTICS": True,
    "LOCAL_MIRROR": "https://builds.session.codes/deps",
}

def cmake_args(opts):
    return " ".join([
        "-D%s=%s" % (k, ("ON" if v else "OFF") if type(v) == "bool" else v)
        for k, v in opts.items()
    ])

def debian(name, image, arch = "amd64", deps = default_deps, cmake = {}, jobs = 6):
    return {
        "name": name,
        "labels": {"platform": "linux/" + arch, "backend": "docker"},
        "when": [{"event": events}],
        "steps": [{
            "name": "build & test",
            "image": registry + image,
            "pull": True,
            "commands": [
                'echo "Building on $${CI_MACHINE}"',
                'echo "man-db man-db/auto-update boolean false" | debconf-set-selections',
                apt_get + " update",
                apt_get + " install -y eatmydata",
                "eatmydata " + apt_get + " dist-upgrade -y",
                "eatmydata " + apt_get + " install --no-install-recommends -y " +
                " ".join(["cmake", "make", "git", "ca-certificates"] + deps),
                "mkdir build",
                "cd build",
                "cmake .. " + cmake_args(dict(default_cmake, **cmake)),
                "make VERBOSE=1 -j%d" % jobs,
                "./tests/testAll --colour-mode ansi -d yes",
            ],
        }],
    }

def main(ctx):
    return [
        debian("Debian sid", "debian-sid"),
    ]
