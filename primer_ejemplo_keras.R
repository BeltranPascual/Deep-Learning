# install.packages("tensorflow")
# install.packages("keras", dependencies=TRUE)

# library(keras)
# install_keras(method = "conda")
#  install_keras(tensorflow="gpu")

# install_keras(tensorflow="2.1.0")

# install_github("rstudio/keras")
library(keras)

# https://keras.rstudio.com/
# https://tensorflow.rstudio.com/reference/keras/install_keras/
# install_keras()



# input layer
inputs <- layer_input(shape = c(784))

# outputs compose input + dense layers
predictions <- inputs %>%
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 10, activation = 'softmax')

# create and compile model
model <- keras_model(inputs = inputs, outputs = predictions)
model %>% compile(
  optimizer = 'rmsprop',
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# ejemplo

set.seed(123)
df <- data.frame(a=rnorm(1000),
                 b=rnorm(1000),
                 c=rnorm(1000))
y=ifelse(df$a>-.075 & df$b<.05 & df$c>0,
         sample(c(0,1),1,prob=c(.1,.9)),
         sample(c(1,0),1,prob=c(.1,.9)))
df <- cbind(y,df)

#split into train/test
test_ind <- seq(1, length(df$y), 5) 
train_x <- data.matrix(df[-test_ind,-1])
train_y <- data.matrix(df[-test_ind, 1])
test_x <-  data.matrix(df[test_ind,-1])
test_y <-  data.matrix(df[test_ind, 1])

library(keras)

# set keras seed
use_session_with_seed(345)

# defining a keras sequential model
model <- keras_model_sequential()

# model architecture
model %>% 
  layer_dense(units = 20, input_shape = ncol(train_x)) %>%
  layer_dropout(rate=0.25)%>%
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 1) %>%
  layer_activation(activation = 'sigmoid')

# compiling the defined model with metric = accuracy and optimiser as adam.
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# fitting the model on the training dataset
model %>% fit(train_x, train_y, 
              epochs = 1000)

# score and produce predictions
score <- model %>% evaluate(test_x, test_y)
pred <- model %>% predict(test_x)


