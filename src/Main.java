public class Main {

    public static void main(String[] args) {
        double[][] input = new double[][]{{0.1, 0.2, 0.3}, {0.9, 1, 0.12}};
        double[][] target = new double[][]{{-0.2, 0.2, 0.8, 0.9}, {-0.3, 0.6, -0.8, 1}};
//        double[][] target = new double[][]{{-0.2, 0.2, 0.8, 0.9}};
//        double[][] input = new double[][]{{0.1, 0.2, 0.3}};
        Matrix mInput = Dense.tensor(input);
        Matrix mTarget = Dense.tensor(target);

        NeuralNetwork neuralNetwork = new NeuralNetwork(0.001);
        neuralNetwork.add(new Dense(3, 3, "leakyrelu"));
        neuralNetwork.add(new Dense(3, 4, "leakyrelu"));
        neuralNetwork.add(new Dense(4, 4, "linear"));

        int i = 0;
        while(true) {
            Matrix out = neuralNetwork.feedforward(mInput);
            Matrix loss = NeuralNetwork.mseLoss(out, mTarget);
            loss.backward();
            neuralNetwork.step();
            i++;
            if(i % 1000 == 0){
                System.out.println("Loss : " +loss.mean());
            }
        }
    }
}