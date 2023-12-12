function noise = white_noise(n, clean)
    variance = var(clean);
    noise = unifrnd(0, variance, [1 n]);
end