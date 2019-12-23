import java.util.ArrayList;

public class NeuralNetwork {
    ArrayList<Dense> layers;
    double learningRate;

    NeuralNetwork(double learningRate) {
        this.learningRate = learningRate;
        layers = new ArrayList<>();
    }

    void add(Dense dense) {
        layers.add(dense);
    }

    public static Matrix mseLoss(Matrix out, Matrix target) {
        Matrix loss;
        loss = out.sub(target).sq();
        return loss;
    }

    Matrix feedforward(Matrix input) {
        Matrix out = input;
        for (Dense dense : layers) {
            out = dense.out(out);
        }
        return out;
    }

    void step() {
        for (Dense dense : layers) {
            dense.step(learningRate);
        }
    }

}
