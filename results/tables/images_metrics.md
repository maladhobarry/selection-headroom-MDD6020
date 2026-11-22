| config | backbone | preprocess | n_pcs | outcome | group | n | events | auroc | cal_intercept | cal_slope | brier | auroc_ci | delta_auroc_vs_gardner | delta_ci |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gardner_reference | - | - | 0 | LB | all | 709 | 229 | 0.552 | -0.001 | 0.944 | 0.216 | [0.508, 0.600] |  |  |
| gardner_reference | - | - | 0 | LB | choice | 396 | 130 | 0.498 | 0.017 | 0.377 | 0.222 | [0.432, 0.560] |  |  |
| gardner_reference | - | - | 0 | LB | no_choice | 209 | 58 | 0.629 | -0.127 | 1.513 | 0.194 | [0.545, 0.712] |  |  |
| gardner_reference | - | - | 0 | LB | choice_S10 | 392 | 130 | 0.499 | 0.035 | 0.435 | 0.222 | [0.438, 0.559] |  |  |
| gardner_reference | - | - | 0 | HA | all | 709 | 256 | 0.579 | -0.002 | 0.967 | 0.225 | [0.539, 0.622] |  |  |
| gardner_reference | - | - | 0 | HA | choice | 396 | 149 | 0.534 | 0.042 | 0.520 | 0.235 | [0.475, 0.597] |  |  |
| gardner_reference | - | - | 0 | HA | no_choice | 209 | 62 | 0.643 | -0.163 | 1.394 | 0.199 | [0.559, 0.719] |  |  |
| gardner_reference | - | - | 0 | HA | choice_S10 | 392 | 149 | 0.536 | 0.062 | 0.576 | 0.235 | [0.480, 0.595] |  |  |
| gardner_reference | - | - | 0 | SS | all | 709 | 311 | 0.608 | -0.000 | 0.948 | 0.236 | [0.570, 0.649] |  |  |
| gardner_reference | - | - | 0 | SS | choice | 396 | 178 | 0.574 | -0.008 | 0.719 | 0.242 | [0.516, 0.631] |  |  |
| gardner_reference | - | - | 0 | SS | no_choice | 209 | 83 | 0.655 | -0.043 | 1.131 | 0.223 | [0.579, 0.732] |  |  |
| gardner_reference | - | - | 0 | SS | choice_S10 | 392 | 178 | 0.577 | 0.014 | 0.768 | 0.242 | [0.521, 0.633] |  |  |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | LB | all | 709 | 229 | 0.515 | 0.001 | 0.174 | 0.219 | [0.470, 0.558] | -0.036 | [-0.076, -0.000] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | LB | choice | 396 | 130 | 0.479 | 0.029 | -1.060 | 0.222 | [0.420, 0.542] | -0.019 | [-0.066, 0.033] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | LB | no_choice | 209 | 58 | 0.549 | -0.188 | 1.760 | 0.201 | [0.463, 0.642] | -0.079 | [-0.156, 0.002] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | LB | choice_S10 | 392 | 130 | 0.483 | 0.045 | -0.916 | 0.223 | [0.425, 0.542] | -0.016 | [-0.066, 0.033] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | HA | all | 709 | 256 | 0.555 | -0.000 | 0.701 | 0.230 | [0.507, 0.598] | -0.025 | [-0.062, 0.012] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | HA | choice | 396 | 149 | 0.515 | 0.058 | 0.136 | 0.237 | [0.456, 0.572] | -0.019 | [-0.069, 0.036] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | HA | no_choice | 209 | 62 | 0.600 | -0.222 | 1.439 | 0.208 | [0.523, 0.689] | -0.043 | [-0.108, 0.033] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | HA | choice_S10 | 392 | 149 | 0.519 | 0.076 | 0.199 | 0.237 | [0.461, 0.578] | -0.018 | [-0.070, 0.034] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | SS | all | 709 | 311 | 0.583 | 0.002 | 0.926 | 0.243 | [0.542, 0.627] | -0.025 | [-0.062, 0.012] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | SS | choice | 396 | 178 | 0.569 | 0.021 | 0.702 | 0.245 | [0.516, 0.623] | -0.005 | [-0.053, 0.043] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | SS | no_choice | 209 | 83 | 0.600 | -0.097 | 1.423 | 0.234 | [0.526, 0.679] | -0.055 | [-0.127, 0.027] |
| resnet50_imagenet|pad|10 | resnet50_imagenet | pad | 10 | SS | choice_S10 | 392 | 178 | 0.572 | 0.041 | 0.723 | 0.246 | [0.519, 0.627] | -0.005 | [-0.055, 0.048] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | LB | all | 709 | 229 | 0.513 | 0.001 | -0.203 | 0.219 | [0.470, 0.553] | -0.038 | [-0.088, 0.003] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | LB | choice | 396 | 130 | 0.487 | 0.030 | -1.223 | 0.222 | [0.428, 0.543] | -0.011 | [-0.067, 0.056] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | LB | no_choice | 209 | 58 | 0.549 | -0.196 | 1.550 | 0.202 | [0.461, 0.636] | -0.079 | [-0.168, 0.012] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | LB | choice_S10 | 392 | 130 | 0.490 | 0.045 | -1.137 | 0.223 | [0.434, 0.546] | -0.010 | [-0.066, 0.055] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | HA | all | 709 | 256 | 0.550 | -0.000 | 0.711 | 0.230 | [0.503, 0.587] | -0.029 | [-0.071, 0.011] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | HA | choice | 396 | 149 | 0.516 | 0.065 | 0.094 | 0.236 | [0.461, 0.572] | -0.018 | [-0.073, 0.041] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | HA | no_choice | 209 | 62 | 0.599 | -0.248 | 2.377 | 0.208 | [0.519, 0.688] | -0.044 | [-0.122, 0.050] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | HA | choice_S10 | 392 | 149 | 0.519 | 0.082 | 0.146 | 0.237 | [0.465, 0.576] | -0.017 | [-0.069, 0.043] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | SS | all | 709 | 311 | 0.586 | 0.003 | 0.852 | 0.243 | [0.546, 0.630] | -0.022 | [-0.063, 0.018] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | SS | choice | 396 | 178 | 0.571 | 0.025 | 0.589 | 0.246 | [0.516, 0.621] | -0.004 | [-0.064, 0.049] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | SS | no_choice | 209 | 83 | 0.612 | -0.101 | 1.640 | 0.233 | [0.536, 0.694] | -0.043 | [-0.126, 0.036] |
| resnet50_imagenet|pad|20 | resnet50_imagenet | pad | 20 | SS | choice_S10 | 392 | 178 | 0.571 | 0.044 | 0.587 | 0.247 | [0.517, 0.623] | -0.007 | [-0.058, 0.047] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | LB | all | 709 | 229 | 0.536 | 0.003 | 0.774 | 0.218 | [0.489, 0.585] | -0.015 | [-0.065, 0.033] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | LB | choice | 396 | 130 | 0.508 | 0.029 | 0.346 | 0.221 | [0.456, 0.568] | 0.010 | [-0.051, 0.079] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | LB | no_choice | 209 | 58 | 0.598 | -0.190 | 2.476 | 0.200 | [0.511, 0.681] | -0.031 | [-0.120, 0.052] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | LB | choice_S10 | 392 | 130 | 0.510 | 0.046 | 0.438 | 0.222 | [0.452, 0.570] | 0.011 | [-0.059, 0.079] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | HA | all | 709 | 256 | 0.553 | -0.000 | 0.935 | 0.229 | [0.505, 0.594] | -0.026 | [-0.074, 0.019] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | HA | choice | 396 | 149 | 0.523 | 0.061 | 0.437 | 0.235 | [0.473, 0.584] | -0.011 | [-0.076, 0.057] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | HA | no_choice | 209 | 62 | 0.608 | -0.259 | 2.155 | 0.208 | [0.514, 0.693] | -0.035 | [-0.124, 0.045] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | HA | choice_S10 | 392 | 149 | 0.526 | 0.079 | 0.508 | 0.236 | [0.472, 0.585] | -0.010 | [-0.070, 0.060] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | SS | all | 709 | 311 | 0.576 | 0.003 | 0.755 | 0.243 | [0.532, 0.616] | -0.031 | [-0.077, 0.011] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | SS | choice | 396 | 178 | 0.561 | 0.028 | 0.535 | 0.247 | [0.507, 0.615] | -0.013 | [-0.077, 0.048] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | SS | no_choice | 209 | 83 | 0.586 | -0.112 | 1.128 | 0.235 | [0.504, 0.668] | -0.069 | [-0.142, 0.006] |
| resnet50_imagenet|pad|50 | resnet50_imagenet | pad | 50 | SS | choice_S10 | 392 | 178 | 0.563 | 0.047 | 0.544 | 0.247 | [0.509, 0.617] | -0.015 | [-0.066, 0.042] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | LB | all | 709 | 229 | 0.538 | 0.001 | 0.520 | 0.219 | [0.494, 0.580] | -0.013 | [-0.056, 0.024] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | LB | choice | 396 | 130 | 0.505 | 0.031 | -0.209 | 0.222 | [0.449, 0.572] | 0.007 | [-0.043, 0.064] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | LB | no_choice | 209 | 58 | 0.599 | -0.173 | 2.122 | 0.199 | [0.507, 0.689] | -0.029 | [-0.107, 0.048] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | LB | choice_S10 | 392 | 130 | 0.507 | 0.047 | -0.163 | 0.223 | [0.453, 0.566] | 0.008 | [-0.043, 0.062] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | HA | all | 709 | 256 | 0.565 | -0.001 | 0.806 | 0.229 | [0.521, 0.607] | -0.014 | [-0.052, 0.022] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | HA | choice | 396 | 149 | 0.529 | 0.058 | 0.322 | 0.236 | [0.471, 0.588] | -0.005 | [-0.052, 0.048] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | HA | no_choice | 209 | 62 | 0.617 | -0.213 | 1.668 | 0.205 | [0.541, 0.701] | -0.026 | [-0.092, 0.052] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | HA | choice_S10 | 392 | 149 | 0.531 | 0.076 | 0.348 | 0.237 | [0.475, 0.592] | -0.005 | [-0.056, 0.048] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | SS | all | 709 | 311 | 0.592 | 0.003 | 1.028 | 0.241 | [0.553, 0.634] | -0.016 | [-0.053, 0.021] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | SS | choice | 396 | 178 | 0.577 | 0.020 | 0.775 | 0.244 | [0.520, 0.630] | 0.002 | [-0.045, 0.047] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | SS | no_choice | 209 | 83 | 0.621 | -0.090 | 1.513 | 0.231 | [0.549, 0.703] | -0.034 | [-0.106, 0.041] |
| resnet50_imagenet|clahe|10 | resnet50_imagenet | clahe | 10 | SS | choice_S10 | 392 | 178 | 0.579 | 0.040 | 0.790 | 0.244 | [0.524, 0.632] | 0.001 | [-0.045, 0.053] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | LB | all | 709 | 229 | 0.536 | 0.002 | 0.408 | 0.219 | [0.489, 0.578] | -0.016 | [-0.064, 0.029] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | LB | choice | 396 | 130 | 0.504 | 0.026 | -0.243 | 0.222 | [0.449, 0.558] | 0.006 | [-0.063, 0.074] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | LB | no_choice | 209 | 58 | 0.592 | -0.178 | 1.371 | 0.200 | [0.511, 0.681] | -0.037 | [-0.114, 0.048] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | LB | choice_S10 | 392 | 130 | 0.507 | 0.042 | -0.153 | 0.223 | [0.446, 0.565] | 0.007 | [-0.056, 0.076] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | HA | all | 709 | 256 | 0.561 | 0.001 | 0.823 | 0.229 | [0.514, 0.603] | -0.019 | [-0.060, 0.022] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | HA | choice | 396 | 149 | 0.532 | 0.058 | 0.305 | 0.236 | [0.477, 0.587] | -0.002 | [-0.061, 0.055] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | HA | no_choice | 209 | 62 | 0.593 | -0.236 | 1.648 | 0.208 | [0.513, 0.676] | -0.050 | [-0.121, 0.027] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | HA | choice_S10 | 392 | 149 | 0.536 | 0.077 | 0.394 | 0.236 | [0.478, 0.593] | -0.001 | [-0.056, 0.066] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | SS | all | 709 | 311 | 0.582 | 0.000 | 0.997 | 0.243 | [0.539, 0.625] | -0.026 | [-0.069, 0.016] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | SS | choice | 396 | 178 | 0.564 | 0.014 | 0.699 | 0.246 | [0.509, 0.614] | -0.010 | [-0.060, 0.038] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | SS | no_choice | 209 | 83 | 0.605 | -0.100 | 1.397 | 0.235 | [0.530, 0.686] | -0.050 | [-0.122, 0.015] |
| resnet50_imagenet|clahe|20 | resnet50_imagenet | clahe | 20 | SS | choice_S10 | 392 | 178 | 0.567 | 0.034 | 0.729 | 0.246 | [0.514, 0.622] | -0.010 | [-0.066, 0.045] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | LB | all | 709 | 229 | 0.505 | 0.002 | 0.058 | 0.219 | [0.459, 0.555] | -0.046 | [-0.097, -0.003] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | LB | choice | 396 | 130 | 0.451 | 0.026 | -1.796 | 0.222 | [0.392, 0.509] | -0.047 | [-0.109, 0.017] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | LB | no_choice | 209 | 58 | 0.578 | -0.193 | 3.112 | 0.201 | [0.494, 0.666] | -0.051 | [-0.137, 0.038] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | LB | choice_S10 | 392 | 130 | 0.453 | 0.041 | -1.706 | 0.223 | [0.394, 0.515] | -0.046 | [-0.108, 0.020] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | HA | all | 709 | 256 | 0.540 | -0.001 | 0.552 | 0.231 | [0.495, 0.586] | -0.039 | [-0.084, 0.004] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | HA | choice | 396 | 149 | 0.499 | 0.059 | -0.458 | 0.237 | [0.439, 0.553] | -0.035 | [-0.094, 0.022] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | HA | no_choice | 209 | 62 | 0.568 | -0.263 | 1.500 | 0.210 | [0.482, 0.659] | -0.075 | [-0.146, -0.003] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | HA | choice_S10 | 392 | 149 | 0.503 | 0.077 | -0.338 | 0.237 | [0.446, 0.563] | -0.034 | [-0.097, 0.030] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | SS | all | 709 | 311 | 0.570 | 0.004 | 0.722 | 0.244 | [0.529, 0.612] | -0.038 | [-0.082, 0.004] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | SS | choice | 396 | 178 | 0.546 | 0.024 | 0.370 | 0.248 | [0.493, 0.600] | -0.028 | [-0.084, 0.034] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | SS | no_choice | 209 | 83 | 0.572 | -0.114 | 0.952 | 0.237 | [0.492, 0.648] | -0.083 | [-0.159, 0.005] |
| resnet50_imagenet|clahe|50 | resnet50_imagenet | clahe | 50 | SS | choice_S10 | 392 | 178 | 0.550 | 0.045 | 0.412 | 0.249 | [0.493, 0.607] | -0.027 | [-0.084, 0.033] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | LB | all | 709 | 229 | 0.533 | -0.001 | 0.717 | 0.218 | [0.487, 0.578] | -0.018 | [-0.066, 0.026] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | LB | choice | 396 | 130 | 0.499 | 0.028 | 0.100 | 0.221 | [0.440, 0.564] | 0.002 | [-0.061, 0.066] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | LB | no_choice | 209 | 58 | 0.574 | -0.163 | 1.936 | 0.199 | [0.495, 0.655] | -0.055 | [-0.149, 0.049] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | LB | choice_S10 | 392 | 130 | 0.502 | 0.044 | 0.155 | 0.223 | [0.445, 0.560] | 0.003 | [-0.060, 0.061] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | HA | all | 709 | 256 | 0.561 | -0.002 | 0.816 | 0.229 | [0.518, 0.605] | -0.018 | [-0.062, 0.022] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | HA | choice | 396 | 149 | 0.542 | 0.058 | 0.573 | 0.234 | [0.487, 0.602] | 0.008 | [-0.057, 0.069] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | HA | no_choice | 209 | 62 | 0.566 | -0.211 | 0.937 | 0.209 | [0.492, 0.642] | -0.077 | [-0.157, 0.014] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | HA | choice_S10 | 392 | 149 | 0.545 | 0.077 | 0.624 | 0.235 | [0.493, 0.598] | 0.009 | [-0.049, 0.072] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | SS | all | 709 | 311 | 0.587 | -0.001 | 0.956 | 0.242 | [0.546, 0.629] | -0.020 | [-0.059, 0.020] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | SS | choice | 396 | 178 | 0.588 | 0.019 | 0.929 | 0.243 | [0.531, 0.646] | 0.013 | [-0.041, 0.067] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | SS | no_choice | 209 | 83 | 0.573 | -0.091 | 0.780 | 0.237 | [0.500, 0.651] | -0.082 | [-0.158, -0.005] |
| dinov2_vits14|pad|10 | dinov2_vits14 | pad | 10 | SS | choice_S10 | 392 | 178 | 0.590 | 0.039 | 0.959 | 0.243 | [0.536, 0.644] | 0.013 | [-0.041, 0.067] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | LB | all | 709 | 229 | 0.529 | 0.001 | 0.592 | 0.219 | [0.485, 0.575] | -0.023 | [-0.068, 0.016] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | LB | choice | 396 | 130 | 0.488 | 0.031 | -0.270 | 0.222 | [0.432, 0.546] | -0.010 | [-0.075, 0.049] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | LB | no_choice | 209 | 58 | 0.574 | -0.172 | 1.555 | 0.200 | [0.489, 0.660] | -0.055 | [-0.145, 0.050] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | LB | choice_S10 | 392 | 130 | 0.490 | 0.047 | -0.231 | 0.223 | [0.433, 0.550] | -0.009 | [-0.069, 0.053] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | HA | all | 709 | 256 | 0.567 | -0.001 | 0.859 | 0.228 | [0.523, 0.610] | -0.012 | [-0.057, 0.025] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | HA | choice | 396 | 149 | 0.546 | 0.059 | 0.616 | 0.234 | [0.494, 0.601] | 0.012 | [-0.049, 0.070] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | HA | no_choice | 209 | 62 | 0.570 | -0.216 | 0.793 | 0.209 | [0.488, 0.651] | -0.073 | [-0.151, 0.024] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | HA | choice_S10 | 392 | 149 | 0.548 | 0.077 | 0.660 | 0.235 | [0.494, 0.609] | 0.012 | [-0.045, 0.080] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | SS | all | 709 | 311 | 0.580 | 0.003 | 0.971 | 0.242 | [0.540, 0.623] | -0.028 | [-0.065, 0.011] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | SS | choice | 396 | 178 | 0.569 | 0.027 | 0.818 | 0.245 | [0.513, 0.626] | -0.006 | [-0.059, 0.047] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | SS | no_choice | 209 | 83 | 0.572 | -0.102 | 0.755 | 0.238 | [0.493, 0.648] | -0.083 | [-0.157, -0.003] |
| dinov2_vits14|pad|20 | dinov2_vits14 | pad | 20 | SS | choice_S10 | 392 | 178 | 0.571 | 0.047 | 0.860 | 0.245 | [0.522, 0.626] | -0.006 | [-0.055, 0.050] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | LB | all | 709 | 229 | 0.527 | 0.000 | 0.670 | 0.218 | [0.476, 0.571] | -0.024 | [-0.076, 0.020] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | LB | choice | 396 | 130 | 0.495 | 0.030 | -0.241 | 0.222 | [0.430, 0.557] | -0.003 | [-0.079, 0.073] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | LB | no_choice | 209 | 58 | 0.557 | -0.179 | 1.259 | 0.201 | [0.475, 0.651] | -0.071 | [-0.166, 0.039] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | LB | choice_S10 | 392 | 130 | 0.494 | 0.045 | -0.262 | 0.223 | [0.432, 0.560] | -0.005 | [-0.078, 0.068] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | HA | all | 709 | 256 | 0.558 | -0.000 | 0.820 | 0.229 | [0.510, 0.599] | -0.021 | [-0.071, 0.019] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | HA | choice | 396 | 149 | 0.550 | 0.064 | 0.624 | 0.234 | [0.496, 0.611] | 0.017 | [-0.050, 0.081] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | HA | no_choice | 209 | 62 | 0.536 | -0.237 | 0.550 | 0.211 | [0.453, 0.618] | -0.107 | [-0.191, -0.014] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | HA | choice_S10 | 392 | 149 | 0.552 | 0.080 | 0.623 | 0.235 | [0.495, 0.619] | 0.016 | [-0.052, 0.088] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | SS | all | 709 | 311 | 0.567 | 0.001 | 0.884 | 0.243 | [0.525, 0.611] | -0.041 | [-0.083, 0.002] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | SS | choice | 396 | 178 | 0.567 | 0.029 | 0.830 | 0.245 | [0.514, 0.624] | -0.007 | [-0.068, 0.051] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | SS | no_choice | 209 | 83 | 0.538 | -0.106 | 0.601 | 0.239 | [0.454, 0.619] | -0.117 | [-0.198, -0.028] |
| dinov2_vits14|pad|50 | dinov2_vits14 | pad | 50 | SS | choice_S10 | 392 | 178 | 0.569 | 0.048 | 0.850 | 0.245 | [0.514, 0.625] | -0.008 | [-0.065, 0.052] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | LB | all | 709 | 229 | 0.551 | -0.002 | 0.797 | 0.218 | [0.501, 0.594] | -0.000 | [-0.053, 0.047] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | LB | choice | 396 | 130 | 0.520 | 0.025 | 0.266 | 0.222 | [0.464, 0.582] | 0.022 | [-0.040, 0.087] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | LB | no_choice | 209 | 58 | 0.616 | -0.148 | 2.187 | 0.197 | [0.530, 0.696] | -0.013 | [-0.111, 0.093] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | LB | choice_S10 | 392 | 130 | 0.525 | 0.042 | 0.337 | 0.222 | [0.465, 0.583] | 0.025 | [-0.032, 0.090] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | HA | all | 709 | 256 | 0.582 | 0.001 | 0.899 | 0.227 | [0.533, 0.627] | 0.003 | [-0.043, 0.044] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | HA | choice | 396 | 149 | 0.553 | 0.051 | 0.563 | 0.234 | [0.497, 0.610] | 0.020 | [-0.043, 0.079] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | HA | no_choice | 209 | 62 | 0.640 | -0.180 | 1.721 | 0.202 | [0.560, 0.719] | -0.003 | [-0.082, 0.085] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | HA | choice_S10 | 392 | 149 | 0.559 | 0.071 | 0.634 | 0.235 | [0.501, 0.616] | 0.022 | [-0.038, 0.089] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | SS | all | 709 | 311 | 0.598 | 0.001 | 0.993 | 0.241 | [0.553, 0.639] | -0.010 | [-0.049, 0.028] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | SS | choice | 396 | 178 | 0.594 | 0.009 | 0.867 | 0.243 | [0.542, 0.646] | 0.019 | [-0.033, 0.076] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | SS | no_choice | 209 | 83 | 0.623 | -0.066 | 1.383 | 0.231 | [0.547, 0.694] | -0.033 | [-0.116, 0.054] |
| dinov2_vits14|clahe|10 | dinov2_vits14 | clahe | 10 | SS | choice_S10 | 392 | 178 | 0.599 | 0.031 | 0.918 | 0.243 | [0.545, 0.652] | 0.021 | [-0.030, 0.079] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | LB | all | 709 | 229 | 0.542 | 0.001 | 0.701 | 0.218 | [0.496, 0.585] | -0.010 | [-0.058, 0.033] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | LB | choice | 396 | 130 | 0.509 | 0.030 | 0.051 | 0.222 | [0.451, 0.566] | 0.011 | [-0.052, 0.072] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | LB | no_choice | 209 | 58 | 0.575 | -0.165 | 1.382 | 0.200 | [0.489, 0.660] | -0.053 | [-0.154, 0.053] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | LB | choice_S10 | 392 | 130 | 0.511 | 0.046 | 0.082 | 0.223 | [0.451, 0.572] | 0.012 | [-0.051, 0.078] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | HA | all | 709 | 256 | 0.577 | -0.001 | 0.914 | 0.227 | [0.532, 0.622] | -0.002 | [-0.051, 0.039] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | HA | choice | 396 | 149 | 0.551 | 0.054 | 0.622 | 0.234 | [0.491, 0.608] | 0.017 | [-0.047, 0.071] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | HA | no_choice | 209 | 62 | 0.599 | -0.200 | 1.190 | 0.206 | [0.518, 0.676] | -0.044 | [-0.122, 0.052] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | HA | choice_S10 | 392 | 149 | 0.553 | 0.072 | 0.653 | 0.235 | [0.498, 0.614] | 0.017 | [-0.041, 0.085] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | SS | all | 709 | 311 | 0.598 | 0.003 | 1.019 | 0.240 | [0.559, 0.642] | -0.010 | [-0.052, 0.029] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | SS | choice | 396 | 178 | 0.587 | 0.017 | 0.840 | 0.243 | [0.536, 0.647] | 0.013 | [-0.042, 0.066] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | SS | no_choice | 209 | 83 | 0.600 | -0.078 | 1.120 | 0.233 | [0.520, 0.680] | -0.056 | [-0.136, 0.028] |
| dinov2_vits14|clahe|20 | dinov2_vits14 | clahe | 20 | SS | choice_S10 | 392 | 178 | 0.590 | 0.037 | 0.869 | 0.243 | [0.542, 0.644] | 0.013 | [-0.038, 0.073] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | LB | all | 709 | 229 | 0.514 | 0.001 | 0.346 | 0.219 | [0.466, 0.560] | -0.038 | [-0.088, 0.003] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | LB | choice | 396 | 130 | 0.476 | 0.031 | -0.802 | 0.222 | [0.415, 0.536] | -0.022 | [-0.096, 0.048] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | LB | no_choice | 209 | 58 | 0.550 | -0.185 | 1.155 | 0.201 | [0.458, 0.644] | -0.078 | [-0.196, 0.035] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | LB | choice_S10 | 392 | 130 | 0.477 | 0.046 | -0.800 | 0.223 | [0.416, 0.542] | -0.023 | [-0.098, 0.047] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | HA | all | 709 | 256 | 0.536 | -0.000 | 0.697 | 0.230 | [0.487, 0.582] | -0.043 | [-0.092, -0.002] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | HA | choice | 396 | 149 | 0.508 | 0.062 | 0.189 | 0.236 | [0.454, 0.568] | -0.026 | [-0.094, 0.038] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | HA | no_choice | 209 | 62 | 0.550 | -0.248 | 0.738 | 0.211 | [0.469, 0.635] | -0.093 | [-0.185, 0.004] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | HA | choice_S10 | 392 | 149 | 0.510 | 0.078 | 0.214 | 0.237 | [0.454, 0.573] | -0.027 | [-0.094, 0.046] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | SS | all | 709 | 311 | 0.569 | 0.002 | 0.960 | 0.244 | [0.529, 0.615] | -0.039 | [-0.081, 0.002] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | SS | choice | 396 | 178 | 0.561 | 0.030 | 0.711 | 0.246 | [0.507, 0.616] | -0.013 | [-0.072, 0.043] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | SS | no_choice | 209 | 83 | 0.550 | -0.113 | 0.839 | 0.238 | [0.470, 0.634] | -0.106 | [-0.192, -0.012] |
| dinov2_vits14|clahe|50 | dinov2_vits14 | clahe | 50 | SS | choice_S10 | 392 | 178 | 0.564 | 0.049 | 0.739 | 0.247 | [0.515, 0.621] | -0.013 | [-0.068, 0.055] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | LB | all | 709 | 229 | 0.533 | -0.001 | 0.618 | 0.218 | [0.488, 0.577] | -0.018 | [-0.062, 0.023] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | LB | choice | 396 | 130 | 0.505 | 0.026 | -0.035 | 0.222 | [0.445, 0.568] | 0.007 | [-0.053, 0.066] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | LB | no_choice | 209 | 58 | 0.577 | -0.167 | 1.625 | 0.200 | [0.489, 0.661] | -0.052 | [-0.139, 0.027] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | LB | choice_S10 | 392 | 130 | 0.507 | 0.042 | 0.066 | 0.222 | [0.451, 0.573] | 0.008 | [-0.051, 0.070] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | HA | all | 709 | 256 | 0.564 | -0.002 | 0.975 | 0.228 | [0.521, 0.606] | -0.015 | [-0.053, 0.024] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | HA | choice | 396 | 149 | 0.538 | 0.051 | 0.573 | 0.234 | [0.481, 0.595] | 0.005 | [-0.052, 0.059] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | HA | no_choice | 209 | 62 | 0.602 | -0.209 | 1.826 | 0.207 | [0.522, 0.685] | -0.041 | [-0.118, 0.042] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | HA | choice_S10 | 392 | 149 | 0.541 | 0.069 | 0.646 | 0.235 | [0.483, 0.603] | 0.004 | [-0.047, 0.061] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | SS | all | 709 | 311 | 0.595 | -0.000 | 1.057 | 0.240 | [0.555, 0.641] | -0.012 | [-0.047, 0.024] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | SS | choice | 396 | 178 | 0.600 | 0.013 | 1.130 | 0.240 | [0.541, 0.657] | 0.026 | [-0.025, 0.080] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | SS | no_choice | 209 | 83 | 0.577 | -0.076 | 0.715 | 0.238 | [0.502, 0.655] | -0.078 | [-0.151, 0.001] |
| resnet50_gardner|pad|10 | resnet50_gardner | pad | 10 | SS | choice_S10 | 392 | 178 | 0.603 | 0.034 | 1.201 | 0.240 | [0.551, 0.658] | 0.026 | [-0.017, 0.083] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | LB | all | 709 | 229 | 0.542 | 0.000 | 0.407 | 0.219 | [0.494, 0.586] | -0.009 | [-0.056, 0.034] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | LB | choice | 396 | 130 | 0.509 | 0.024 | -0.273 | 0.222 | [0.443, 0.575] | 0.011 | [-0.051, 0.068] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | LB | no_choice | 209 | 58 | 0.587 | -0.166 | 1.469 | 0.200 | [0.493, 0.677] | -0.042 | [-0.131, 0.063] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | LB | choice_S10 | 392 | 130 | 0.511 | 0.041 | -0.177 | 0.223 | [0.450, 0.569] | 0.012 | [-0.047, 0.072] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | HA | all | 709 | 256 | 0.580 | -0.002 | 0.949 | 0.227 | [0.534, 0.623] | 0.001 | [-0.044, 0.045] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | HA | choice | 396 | 149 | 0.542 | 0.050 | 0.463 | 0.235 | [0.482, 0.606] | 0.009 | [-0.054, 0.067] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | HA | no_choice | 209 | 62 | 0.631 | -0.203 | 1.865 | 0.205 | [0.547, 0.721] | -0.012 | [-0.089, 0.077] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | HA | choice_S10 | 392 | 149 | 0.545 | 0.070 | 0.567 | 0.235 | [0.489, 0.602] | 0.009 | [-0.047, 0.064] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | SS | all | 709 | 311 | 0.606 | 0.001 | 0.993 | 0.239 | [0.567, 0.652] | -0.001 | [-0.041, 0.040] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | SS | choice | 396 | 178 | 0.602 | 0.010 | 0.899 | 0.241 | [0.544, 0.658] | 0.027 | [-0.027, 0.081] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | SS | no_choice | 209 | 83 | 0.594 | -0.069 | 0.934 | 0.235 | [0.513, 0.675] | -0.062 | [-0.138, 0.014] |
| resnet50_gardner|pad|20 (PRIMARY) | resnet50_gardner | pad | 20 | SS | choice_S10 | 392 | 178 | 0.605 | 0.032 | 0.995 | 0.241 | [0.551, 0.663] | 0.028 | [-0.022, 0.087] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | LB | all | 709 | 229 | 0.499 | -0.000 | -0.575 | 0.219 | [0.451, 0.546] | -0.052 | [-0.100, -0.002] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | LB | choice | 396 | 130 | 0.478 | 0.025 | -1.533 | 0.222 | [0.415, 0.540] | -0.020 | [-0.086, 0.045] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | LB | no_choice | 209 | 58 | 0.488 | -0.194 | -0.948 | 0.203 | [0.388, 0.587] | -0.141 | [-0.237, -0.038] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | LB | choice_S10 | 392 | 130 | 0.482 | 0.042 | -1.348 | 0.223 | [0.421, 0.546] | -0.018 | [-0.080, 0.045] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | HA | all | 709 | 256 | 0.544 | 0.002 | 0.758 | 0.230 | [0.501, 0.589] | -0.035 | [-0.085, 0.010] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | HA | choice | 396 | 149 | 0.511 | 0.064 | -0.022 | 0.236 | [0.453, 0.572] | -0.022 | [-0.082, 0.045] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | HA | no_choice | 209 | 62 | 0.561 | -0.253 | 1.530 | 0.210 | [0.476, 0.648] | -0.082 | [-0.159, 0.012] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | HA | choice_S10 | 392 | 149 | 0.516 | 0.082 | 0.137 | 0.237 | [0.456, 0.574] | -0.021 | [-0.084, 0.038] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | SS | all | 709 | 311 | 0.573 | 0.002 | 1.002 | 0.244 | [0.531, 0.617] | -0.034 | [-0.078, 0.009] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | SS | choice | 396 | 178 | 0.573 | 0.030 | 0.879 | 0.245 | [0.514, 0.630] | -0.001 | [-0.055, 0.062] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | SS | no_choice | 209 | 83 | 0.539 | -0.122 | 0.589 | 0.240 | [0.463, 0.618] | -0.116 | [-0.191, -0.034] |
| resnet50_gardner|pad|50 | resnet50_gardner | pad | 50 | SS | choice_S10 | 392 | 178 | 0.579 | 0.052 | 1.053 | 0.245 | [0.521, 0.635] | 0.001 | [-0.050, 0.070] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | LB | all | 709 | 229 | 0.508 | 0.000 | 0.108 | 0.219 | [0.464, 0.552] | -0.043 | [-0.087, -0.002] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | LB | choice | 396 | 130 | 0.493 | 0.030 | -0.417 | 0.221 | [0.431, 0.558] | -0.005 | [-0.062, 0.058] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | LB | no_choice | 209 | 58 | 0.512 | -0.189 | 0.364 | 0.202 | [0.426, 0.596] | -0.117 | [-0.202, -0.038] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | LB | choice_S10 | 392 | 130 | 0.497 | 0.046 | -0.272 | 0.222 | [0.439, 0.559] | -0.003 | [-0.061, 0.062] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | HA | all | 709 | 256 | 0.554 | 0.000 | 0.727 | 0.230 | [0.513, 0.596] | -0.025 | [-0.065, 0.015] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | HA | choice | 396 | 149 | 0.546 | 0.063 | 0.543 | 0.235 | [0.490, 0.608] | 0.012 | [-0.045, 0.074] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | HA | no_choice | 209 | 62 | 0.553 | -0.227 | 0.712 | 0.211 | [0.468, 0.643] | -0.090 | [-0.170, -0.003] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | HA | choice_S10 | 392 | 149 | 0.551 | 0.082 | 0.628 | 0.235 | [0.500, 0.611] | 0.014 | [-0.041, 0.074] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | SS | all | 709 | 311 | 0.580 | 0.003 | 0.938 | 0.242 | [0.541, 0.624] | -0.028 | [-0.062, 0.009] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | SS | choice | 396 | 178 | 0.596 | 0.025 | 1.214 | 0.241 | [0.541, 0.650] | 0.021 | [-0.031, 0.079] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | SS | no_choice | 209 | 83 | 0.540 | -0.092 | 0.249 | 0.242 | [0.464, 0.624] | -0.115 | [-0.183, -0.039] |
| resnet50_gardner|clahe|10 | resnet50_gardner | clahe | 10 | SS | choice_S10 | 392 | 178 | 0.600 | 0.046 | 1.301 | 0.241 | [0.552, 0.656] | 0.023 | [-0.024, 0.080] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | LB | all | 709 | 229 | 0.503 | 0.001 | -0.280 | 0.219 | [0.453, 0.549] | -0.048 | [-0.097, -0.006] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | LB | choice | 396 | 130 | 0.476 | 0.030 | -1.452 | 0.222 | [0.417, 0.536] | -0.022 | [-0.085, 0.046] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | LB | no_choice | 209 | 58 | 0.535 | -0.196 | 1.166 | 0.202 | [0.446, 0.629] | -0.094 | [-0.185, -0.001] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | LB | choice_S10 | 392 | 130 | 0.479 | 0.045 | -1.313 | 0.223 | [0.427, 0.544] | -0.021 | [-0.083, 0.043] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | HA | all | 709 | 256 | 0.561 | 0.001 | 0.906 | 0.229 | [0.512, 0.605] | -0.018 | [-0.063, 0.022] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | HA | choice | 396 | 149 | 0.537 | 0.066 | 0.455 | 0.235 | [0.475, 0.598] | 0.003 | [-0.058, 0.070] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | HA | no_choice | 209 | 62 | 0.600 | -0.239 | 1.553 | 0.209 | [0.510, 0.687] | -0.043 | [-0.128, 0.047] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | HA | choice_S10 | 392 | 149 | 0.540 | 0.085 | 0.531 | 0.236 | [0.487, 0.606] | 0.003 | [-0.055, 0.070] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | SS | all | 709 | 311 | 0.563 | 0.004 | 0.903 | 0.244 | [0.521, 0.605] | -0.044 | [-0.081, -0.010] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | SS | choice | 396 | 178 | 0.568 | 0.033 | 1.047 | 0.245 | [0.514, 0.622] | -0.007 | [-0.059, 0.050] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | SS | no_choice | 209 | 83 | 0.532 | -0.118 | 0.190 | 0.242 | [0.451, 0.615] | -0.124 | [-0.193, -0.042] |
| resnet50_gardner|clahe|20 | resnet50_gardner | clahe | 20 | SS | choice_S10 | 392 | 178 | 0.571 | 0.053 | 1.132 | 0.245 | [0.522, 0.630] | -0.006 | [-0.053, 0.053] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | LB | all | 709 | 229 | 0.489 | 0.001 | -0.680 | 0.219 | [0.441, 0.534] | -0.063 | [-0.111, -0.017] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | LB | choice | 396 | 130 | 0.474 | 0.026 | -1.215 | 0.222 | [0.415, 0.539] | -0.023 | [-0.091, 0.042] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | LB | no_choice | 209 | 58 | 0.491 | -0.196 | -0.678 | 0.203 | [0.390, 0.581] | -0.137 | [-0.234, -0.038] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | LB | choice_S10 | 392 | 130 | 0.477 | 0.042 | -1.111 | 0.223 | [0.425, 0.537] | -0.022 | [-0.085, 0.046] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | HA | all | 709 | 256 | 0.547 | 0.003 | 0.736 | 0.230 | [0.502, 0.592] | -0.033 | [-0.078, 0.011] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | HA | choice | 396 | 149 | 0.532 | 0.065 | 0.462 | 0.235 | [0.472, 0.593] | -0.002 | [-0.062, 0.064] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | HA | no_choice | 209 | 62 | 0.549 | -0.249 | 0.628 | 0.211 | [0.454, 0.635] | -0.094 | [-0.195, 0.002] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | HA | choice_S10 | 392 | 149 | 0.536 | 0.083 | 0.529 | 0.236 | [0.482, 0.603] | -0.001 | [-0.063, 0.070] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | SS | all | 709 | 311 | 0.557 | 0.002 | 0.853 | 0.245 | [0.515, 0.599] | -0.051 | [-0.094, -0.009] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | SS | choice | 396 | 178 | 0.556 | 0.033 | 0.752 | 0.246 | [0.500, 0.612] | -0.018 | [-0.079, 0.040] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | SS | no_choice | 209 | 83 | 0.523 | -0.132 | 0.386 | 0.241 | [0.440, 0.602] | -0.132 | [-0.210, -0.049] |
| resnet50_gardner|clahe|50 | resnet50_gardner | clahe | 50 | SS | choice_S10 | 392 | 178 | 0.561 | 0.053 | 0.846 | 0.247 | [0.510, 0.620] | -0.016 | [-0.071, 0.049] |
