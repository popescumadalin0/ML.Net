using Microsoft.ML.Data;

namespace ML.Net.Models;

public class SentimentData
{
    [LoadColumn(0)] public string? Text;
    [LoadColumn(1), ColumnName("Label")] public bool Sentiment;
}