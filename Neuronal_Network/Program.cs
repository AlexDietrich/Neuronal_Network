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
            NeuronalNetwork.WriteInFileThread.Join();
            Console.WriteLine("Press any key for exit!");
            Console.ReadKey();
        }
    }
}
