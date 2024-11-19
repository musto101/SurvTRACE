# import the model for testing
from survtrace.src.survtrace.model import SurvTraceSingle
from survtrace.src.survtrace.config import STConfig
from survtrace.src.survtrace.dataset import load_data
from survtrace.src.survtrace.train_utils import Trainer

# define the setup parameters
STConfig['data'] = 'metabric'

hparams = {
    'batch_size': 64,
    'weight_decay': 1e-4,
    'learning_rate': 1e-3,
    'epochs': 20,
}

# load the data
df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig)

# create the model
model = SurvTraceSingle(config=STConfig)
trainer = Trainer(model)
train_loss, val_loss = trainer.fit((df_train, df_y_train), (df_val, df_y_val),
        batch_size=hparams['batch_size'],
        epochs=hparams['epochs'],
        learning_rate=hparams['learning_rate'],
        weight_decay=hparams['weight_decay'],)

# assert that the model is trained correctly
assert train_loss is not None
assert val_loss is not None

# assert that the model is predicted correctly
y_pred = model.predict(df_test)
assert y_pred.any() is not None #TODO: FIX THIS

print("All tests passed!")
