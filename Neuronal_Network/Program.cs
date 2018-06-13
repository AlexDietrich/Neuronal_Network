using System;

namespace Neuronal_Network
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuronalNetwork.ReadDataFromFile();
            NeuronalNetwork.Train();
            NeuronalNetwork.Test();
            //Check if the threads are started or not.. if they are running - join before program exit
            if (NeuronalNetwork.WriteHiddenWeightsFileThread.IsAlive &&
                NeuronalNetwork.WriteInputWeightsFileThread.IsAlive)
            {
                NeuronalNetwork.WriteHiddenWeightsFileThread.Join();
                NeuronalNetwork.WriteInputWeightsFileThread.Join();
            }

            Console.WriteLine("Press any key for exit!");
            Console.ReadKey();
        }
    }
}
