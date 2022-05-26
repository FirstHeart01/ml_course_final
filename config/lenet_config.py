config = dict(
    model_config=dict(
        model_name='MyLeNet',
        loss_type='CrossEntropyLoss',
        loss_weight=[1.14, 1, 1.13, 1.10, 1.15, 1.24, 1.14, 1.08, 1.15, 1.13],
    ),
    data_config=dict(
        random_seed=1,
        batch_size_train=64,
        batch_size_test=64,
        num_workers=0,
        epochs=20,
        test=dict(
            metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
            metric_options=dict(
                topk=(1,),
                thrs=None,
                average_mode='macro'
            ),
        ),
    ),
    optimizer_config=dict(
        type='SGD',
        lr=0.05,
        momentum=0.9,
        weight_decay=1e-4,
    ),
)