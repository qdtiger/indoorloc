"""Tests for config system."""
import pytest
import tempfile
import os


class TestConfig:
    """Tests for Config class."""

    def test_load_yaml(self):
        """Test loading YAML config file."""
        from indoorloc.utils.config import Config

        yaml_content = """
model:
  type: KNNLocalizer
  k: 5
  weights: distance
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name

        try:
            cfg = Config.fromfile(config_path)

            assert cfg.model.type == 'KNNLocalizer'
            assert cfg.model.k == 5
            assert cfg.model.weights == 'distance'
        finally:
            os.unlink(config_path)

    def test_config_inheritance(self):
        """Test config inheritance with _base_."""
        from indoorloc.utils.config import Config

        # Create base config
        base_content = """
model:
  type: KNNLocalizer
  k: 3
  weights: uniform

dataset:
  type: UJIndoorLocDataset
  data_root: data/
"""
        # Create child config
        child_content = """
_base_:
  - base.yaml

model:
  k: 7
  weights: distance
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, 'base.yaml')
            child_path = os.path.join(tmpdir, 'child.yaml')

            with open(base_path, 'w') as f:
                f.write(base_content)

            with open(child_path, 'w') as f:
                f.write(child_content)

            cfg = Config.fromfile(child_path)

            # Check inheritance worked
            assert cfg.model.type == 'KNNLocalizer'  # From base
            assert cfg.model.k == 7  # Overridden
            assert cfg.model.weights == 'distance'  # Overridden
            assert cfg.dataset.type == 'UJIndoorLocDataset'  # From base

    def test_config_to_dict(self):
        """Test converting config to dict."""
        from indoorloc.utils.config import Config

        yaml_content = """
model:
  type: KNNLocalizer
  k: 5
nested:
  level1:
    level2: value
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name

        try:
            cfg = Config.fromfile(config_path)
            d = cfg.to_dict()

            assert isinstance(d, dict)
            assert d['model']['type'] == 'KNNLocalizer'
            assert d['nested']['level1']['level2'] == 'value'
        finally:
            os.unlink(config_path)


class TestRegistry:
    """Tests for Registry system."""

    def test_register_module(self):
        """Test registering a module."""
        from indoorloc.registry import Registry

        test_registry = Registry('test')

        @test_registry.register_module()
        class TestClass:
            def __init__(self, value=1):
                self.value = value

        assert 'TestClass' in test_registry._modules

    def test_build_module(self):
        """Test building module from config."""
        from indoorloc.registry import Registry

        test_registry = Registry('test')

        @test_registry.register_module()
        class TestClass:
            def __init__(self, value=1):
                self.value = value

        cfg = {'type': 'TestClass', 'value': 42}
        instance = test_registry.build(cfg)

        assert instance.value == 42

    def test_register_with_name(self):
        """Test registering with custom name."""
        from indoorloc.registry import Registry

        test_registry = Registry('test')

        @test_registry.register_module(name='custom_name')
        class TestClass:
            pass

        assert 'custom_name' in test_registry._modules
        assert 'TestClass' not in test_registry._modules


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
