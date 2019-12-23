public class Dense {
    int input;
    int output;
    Matrix dense;
    Matrix biases;
    String activate;

    Dense(int input, int output, String transferFunction) {
        this.input = input;
        this.output = output;
        this.activate = transferFunction;
        dense = Matrix.rand(input, output);
        biases = Matrix.rand(output, 1);
    }


    Matrix out(Matrix feed) {
        Matrix out;
        out = feed.dot(dense).T().addSingle(biases).T();
        for (int j = 0; j < out.contents.length; j++) {
            for (int i = 0; i < out.contents[0].length; i++) {
                out.contents[j][i] = activate(activate, out.contents[j][i]);
            }
        }

        return out;
    }

    static Matrix tensor(double[][] array) {
        Tensor[][] matrix = new Tensor[array.length][array[0].length];

        for (int j = 0; j < array.length; j++) {
            for (int i = 0; i < array[0].length; i++) {
                matrix[j][i] = Tensor.variable(array[j][i]);
            }
        }
        return new Matrix(matrix);
    }

    void step(double lr) {
        for (int j = 0; j < dense.rows; j++) {
            for (int i = 0; i < dense.columns; i++) {
                dense.contents[j][i] = Tensor.variable(dense.contents[j][i].getData() - lr * dense.contents[j][i].grad());
            }
            biases.contents[j][0] = Tensor.variable(biases.contents[j][0].getData() - lr * biases.contents[j][0].grad());
        }
    }


    static Tensor activate(String label, Tensor tensor) {

        switch (label) {
            case "gaussian":
                return null;
            case "linear":
                return Tensor.linear(tensor);
            case "sigmoid":
                return Tensor.sigmoid(tensor);
            case "tanh":
                return Tensor.tanh(tensor);
            case "relu":
                return Tensor.relu(tensor);
            case "leakyrelu":
                return Tensor.leakyRelu(tensor);
        }
        throw new Error("Invalid function name: " + label);
    }
}
