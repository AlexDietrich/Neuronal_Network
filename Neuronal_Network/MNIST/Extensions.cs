using System;
using System.IO;

namespace BIF4_MLE_NeuronalNetwork.utils
{
    /// <summary>
    /// Kind regards to Stackoverflow user "koryakinp"
    /// https://stackoverflow.com/questions/49407772/reading-mnist-database
    /// </summary>
    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (var w = 0; w < source.GetLength(0); w++)
            {
                for (var h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
    }
}
