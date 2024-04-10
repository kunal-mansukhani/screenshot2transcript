import tensorflow as tf

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1.0, num_classes=3, name="dice_loss"):
        super(DiceLoss, self).__init__(name=name)
        self.smooth = smooth
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        # One-hot encode y_true
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)
        
        # Apply softmax to y_pred to get probability distributions
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        # Calculate Dice Loss for each class
        intersection = tf.reduce_sum(y_pred * y_true_one_hot, axis=[1, 2])
        union = tf.reduce_sum(y_pred, axis=[1, 2]) + tf.reduce_sum(y_true_one_hot, axis=[1, 2])
        
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - tf.reduce_mean(dice_score)  # Average across classes and batch

        return dice_loss
