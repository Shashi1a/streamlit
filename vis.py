
layer_op = [layer.output for layer in model.layers]
 activation_model = tf.keras.models.Model(
      inputs=model.input, outputs=layer_op)

  fig = plt.figure()

   for x in range(4):
        f1 = activation_model.predict(np.expand_dims(data[0], axis=0))[x]
        ax = fig.add_subplot(1, 4, (x+1))
        # print(f1[0, :, :, 1])
        plt.xticks([])
        plt.yticks([])
        plt.imshow(f1[0, :, :, 1])
    st.pyplot(fig)
