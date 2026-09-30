import socket

from scripts.wait_for import wait_for


def test_open_port_is_ready_closed_one_times_out():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        open_target = f"127.0.0.1:{server.getsockname()[1]}"
        with socket.socket() as probe:  # a port that nothing listens on
            probe.bind(("127.0.0.1", 0))
            closed_target = f"127.0.0.1:{probe.getsockname()[1]}"
        missing = wait_for([open_target, closed_target], timeout_s=0.3, poll_s=0.1)
    assert missing == [closed_target]
