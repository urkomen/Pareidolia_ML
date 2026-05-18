"""
Módulo utils - Funciones auxiliares para el proyecto Pareidolia ML
"""

from .constants import *
from .analysis import (
    renombrar_imagenes_cara,
    renombrar_imagenes_sincara,
    renombrar_predicciones,
    list_images,
    sample_images,
    load_pil,
    analyze_sizes,
    mean_color,
    show_random_by_class,
    symmetry_score,
    find_outliers
)
from .data_loader import (
    read_data,
    load_and_prepare_data,
    save_data_npz,
    load_data_npz,
    preprocess_image
)
from .model_builder import (
    build_model,
    build_efficient_net_b0,
    build_resnet50,
    build_xception,
    build_inception_v3,
    build_vgg16,
    freeze_backbone,
    unfreeze_backbone_layers,
    augmentar_dataset
)
from .training import (
    train_model,
    train_model_custom,
    train_with_augmentation
)
from .evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_roc_curve,
    compare_models_roc,
    summary_table,
    find_optimal_threshold,
    plot_probability_distribution
)
from .prediction import (
    predict_single_image,
    predict_and_visualize,
    batch_predict,
    predict_array,
    get_predictions_summary,
    visualize_predictions,
    prediccion,
    predicciones
)

__all__ = [
    # Constantes
    'IMAGE_WIDTH', 'IMAGE_HEIGHT', 'IMAGE_CHANNELS', 'IMAGE_SIZE',
    'BATCH_SIZE', 'EPOCHS', 'VALIDATION_SPLIT',
    'CATEGORIES', 'CATEGORY_TO_ID',
    'ROOT_PATH', 'TRAIN_PATH', 'TEST_PATH', 'PREDICT_PATH',
    'MODEL_PATH', 'PRODUCTION_MODEL_PATH',
    'DATA_NPZ_PATH', 'DATA_GRAY_NPZ_PATH',
    'FINETUNE_LAYERS', 'FINETUNE_LR', 'FINETUNE_EPOCHS',
    'PREDICTION_THRESHOLD',
    # Data loading
    'read_data',
    'load_and_prepare_data',
    'save_data_npz',
    'load_data_npz',
    'preprocess_image',
    # Model building
    'build_model',
    'build_efficient_net_b0',
    'build_resnet50',
    'build_xception',
    'build_inception_v3',
    'build_vgg16',
    'freeze_backbone',
    'unfreeze_backbone_layers',
    # Training
    'train_model',
    'train_model_custom',
    'train_with_augmentation',
    # Evaluation
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_learning_curves',
    'plot_roc_curve',
    'compare_models_roc',
    'summary_table',
    'find_optimal_threshold',
    'plot_probability_distribution',
    # Prediction
    'predict_single_image',
    'predict_and_visualize',
    'batch_predict',
    'predict_array',
    'get_predictions_summary',
    'visualize_predictions'
]
