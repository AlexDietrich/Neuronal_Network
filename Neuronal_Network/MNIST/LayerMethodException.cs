using System;

namespace BIF4_MLE_NeuronalNetwork.utils
{
    public class LayerMethodException : Exception
    {
        public LayerMethodException()
        {
        }

        public LayerMethodException(string message)
        : base(message)
    {
        }

        public LayerMethodException(string message, Exception inner)
        : base(message, inner)
    {
        }
    }
}
