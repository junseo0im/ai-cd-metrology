from ai_cd_metrology.data_audit import classify_sub_dataset, parse_path_metadata


def test_parse_canonical_fail_image_metadata() -> None:
    metadata = parse_path_metadata(
        "images/images/FAIL/pattern_narrowing/w3_31_rb_pn.png"
    )

    assert metadata == {
        "wafer_id": "w3",
        "die_id": "31",
        "pattern_position": "right-bottom",
        "status": "FAIL",
        "defect_label": "pattern_narrowing",
        "exposure_condition": None,
        "metadata_notes": None,
    }


def test_parse_pass_folder_without_guessing_exposure() -> None:
    metadata = parse_path_metadata("images/images/PASS_normal/w1_11_c.png")

    assert metadata["wafer_id"] == "w1"
    assert metadata["die_id"] == "11"
    assert metadata["pattern_position"] == "center"
    assert metadata["status"] == "PASS"
    assert metadata["defect_label"] == "normal"
    assert metadata["exposure_condition"] is None


def test_conflicting_exposure_tokens_are_not_guessed() -> None:
    metadata = parse_path_metadata(
        "웨이퍼 사진/노광 70/wafer1-E60/noncanonical.png"
    )

    assert metadata["wafer_id"] == "w1"
    assert metadata["exposure_condition"] is None
    assert metadata["metadata_notes"] == "conflicting exposure tokens: 60, 70"


def test_unknown_filename_remains_unknown() -> None:
    metadata = parse_path_metadata("misc/screenshots/example.png")

    assert metadata["wafer_id"] is None
    assert metadata["die_id"] is None
    assert metadata["pattern_position"] is None
    assert metadata["status"] is None
    assert metadata["defect_label"] is None
    assert metadata["exposure_condition"] is None


def test_sub_dataset_classification_uses_stable_folder_depth() -> None:
    assert classify_sub_dataset(
        "images/images/FAIL/particle/w1_13_c_p.png"
    ) == ("images", "images/images/FAIL/particle")
    assert classify_sub_dataset(
        "웨이퍼 사진/노광 70/wafer3/example.png"
    ) == ("웨이퍼 사진", "웨이퍼 사진/노광 70/wafer3")

