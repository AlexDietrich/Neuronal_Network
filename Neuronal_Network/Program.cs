using System;

namespace Neuronal_Network
{
    class Program
    {
        static void Main(string[] args)
        {
            NeuronalNetwork.Train();
            NeuronalNetwork.Test();
            Console.WriteLine("Press any key for exit!");
            Console.ReadKey();
        }
    }
}
