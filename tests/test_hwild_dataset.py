"""Tests for H-WILD dataset wiring (download/layout independent)."""


class TestHWILDDataset:
    def test_dataset_class_import(self):
        from indoorloc.datasets import HWILDDataset, HWILD

        assert HWILDDataset is not None
        assert HWILD is not None
        assert callable(HWILD)

    def test_check_exists_with_env_csv(self, tmp_path):
        from indoorloc.datasets.hwild import HWILDDataset

        ds = HWILDDataset.__new__(HWILDDataset)
        ds.data_root = tmp_path

        assert ds._check_exists() is False

        (tmp_path / 'conference.csv').write_text(
            "uwb_coordinate_x,uwb_coordinate_y,csi_0\n0,0,0\n",
            encoding="utf-8",
        )
        assert ds._check_exists() is True

    def test_load_from_environment_csvs(self, tmp_path):
        from indoorloc.datasets.hwild import HWILDDataset

        header = ['uwb_coordinate_x', 'uwb_coordinate_y'] + [f'csi_{i}' for i in range(90)]
        lines = [','.join(header)]
        for i in range(10):
            row = [str(i), str(i + 0.5)] + [str(float(j)) for j in range(90)]
            lines.append(','.join(row))

        (tmp_path / 'conference.csv').write_text('\n'.join(lines), encoding="utf-8")

        ds = HWILDDataset.__new__(HWILDDataset)
        ds.data_root = tmp_path
        ds.split = 'train'
        ds.environment = 'conference'
        ds.train_ratio = 0.7
        ds._signals = []
        ds._locations = []
        ds._metadata = []

        ds._load_data()

        assert len(ds._signals) == int(10 * 0.7)
        assert len(ds._locations) == len(ds._signals)
        assert len(ds._metadata) == len(ds._signals)
        assert ds._signals[0].to_numpy().shape[0] == HWILDDataset.NUM_FEATURES

