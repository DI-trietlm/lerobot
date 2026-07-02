import pickle

import pytest

from lerobot.rtc_inference.configs import PolicyServerConfig
from lerobot.rtc_inference.helpers import RemotePolicyConfig
from lerobot.rtc_inference import policy_server as policy_server_module
from lerobot.transport import services_pb2


class _DummyContext:
    def peer(self):
        return "test-client"


class _DummyAuditWriter:
    def __init__(self, file_path, *args, **kwargs):
        self.file_path = file_path

    def write_row(self, row):
        pass

    def close(self):
        pass


def test_rtc_policy_server_config_does_not_accept_harness_config():
    with pytest.raises(TypeError):
        PolicyServerConfig.from_dict({"host": "localhost", "port": 9999, "harness": {}})


def test_rtc_policy_server_requires_harness_config_from_client(monkeypatch):
    monkeypatch.setattr(policy_server_module, "_OrderedCsvAuditWriter", _DummyAuditWriter)
    server = policy_server_module.PolicyServer(PolicyServerConfig(host="localhost", port=9999))
    policy_specs = RemotePolicyConfig(
        policy_type="smolvla",
        pretrained_name_or_path="dummy/model",
        lerobot_features={},
        actions_per_chunk=1,
    )
    delattr(policy_specs, "harness_config")
    request = services_pb2.PolicySetup(data=pickle.dumps(policy_specs))

    try:
        with pytest.raises(ValueError, match="harness_config is required"):
            server.SendPolicyInstructions(request, _DummyContext())
    finally:
        server.stop()
