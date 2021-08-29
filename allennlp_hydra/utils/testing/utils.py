from numpy.testing import assert_allclose


def assert_models_weights_equal(model_1, model_2):
    model_1_state_keys = model_1.state_dict().keys()
    model_2_state_keys = model_2.state_dict().keys()
    assert model_1_state_keys == model_2_state_keys
    for key in model_1_state_keys:
        assert_allclose(
            model_1.state_dict()[key].cpu().numpy(),
            model_2.state_dict()[key].cpu().numpy(),
            err_msg=key,
        )
