using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
