from .body_models import SMPLXLayer


smplx_layer_neutral = SMPLXLayer(
    model_path='model_files/smplx/SMPLX_NEUTRAL.npz',
    num_betas=10,
    use_pca=False,
    age='adult',
    kid_template_path='model_files/smplx_kid_template.npy',
)
# smplx_faces = torch.from_numpy(smplx_layer.faces.astype(np.int32))

smplx_layer_male = SMPLXLayer(
    model_path='model_files/smplx/SMPLX_MALE.npz',
    num_betas=10,
    use_pca=False,
    age='adult',
    kid_template_path='model_files/smplx_kid_template.npy',
)

smplx_layer_female = SMPLXLayer(
    model_path='model_files/smplx/SMPLX_FEMALE.npz',
    num_betas=10,
    use_pca=False,
    age='adult',
    kid_template_path='model_files/smplx_kid_template.npy',
)

smplx_layer_neutral_kid = SMPLXLayer(
    model_path='model_files/smplx/SMPLX_NEUTRAL.npz',
    num_betas=10,
    use_pca=False,
    age='kid',
    kid_template_path='model_files/smplx_kid_template.npy',
)
# smplx_faces = torch.from_numpy(smplx_layer.faces.astype(np.int32))

smplx_layer_male_kid = SMPLXLayer(
    model_path='model_files/smplx/SMPLX_MALE.npz',
    num_betas=10,
    use_pca=False,
    age='kid',
    kid_template_path='model_files/smplx_kid_template.npy',
)

smplx_layer_female_kid = SMPLXLayer(
    model_path='model_files/smplx/SMPLX_FEMALE.npz',
    num_betas=10,
    use_pca=False,
    age='kid',
    kid_template_path='model_files/smplx_kid_template.npy',
)


def load_models():
    return (
        smplx_layer_neutral,
        smplx_layer_male,
        smplx_layer_female,
        smplx_layer_neutral_kid,
        smplx_layer_male_kid,
        smplx_layer_female_kid
    )
