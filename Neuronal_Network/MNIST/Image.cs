using System;

namespace BIF4_MLE_NeuronalNetwork.utils
{
    /// <summary>
    /// Kind regards to Stackoverflow user "koryakinp"
    /// https://stackoverflow.com/questions/49407772/reading-mnist-database
    /// </summary>
    public class Image
    {
        public double Label { get; set; }
        public double[,] Data { get; set; }
    }
}
