%% Global Variables%
N = 30;
m = 21;
n = 10;
k = 5;

%% Build Signals%
clean = make_signal(n);
%noise = white_noise(n, clean);
noise = wgn(1, n, 0);
signal = clean + noise;
clean_data_vec = [clean, make_signal(m - 1)];
%noise_data_vec = [noise, white_noise(m - 1, clean_data_vec)];
noise_data_vec = [noise, wgn(1, m - 1, 0)];
signal_data_vec = clean_data_vec + noise_data_vec;

%% Hankel Matrices%%
H_bar = hankel(clean_data_vec);
E = hankel(noise_data_vec);
H = H_bar + E;

SNR = 20*log((norm(clean)) / norm(noise));

%% Covariance matrix estimates %%
[U_bar, S_bar, V_bar] = svd(H_bar);
[U, S, V] = svd(H);

%% Block Matrices %%
U1_bar = U_bar(:, 1:k);
U2_bar = U_bar(:, 1:(N-k));
U1 = U(:, 1:k);
U2 = U(:, 1:(N-k));
V1_bar = V_bar(:, 1:k);
V2_bar = V_bar(:, 1:(N-k));
V1 = V(:, 1:k);
V2 = V(:, 1:(N-k));
S1_bar = S_bar(1:k, 1:k);
S2_bar = zeros(N-k, N-k);
S1 = S(1:k, 1:k);
S2 = S(k+1:N, k+1:N);

%% Variance %%
var_noise = var(noise);

%% Covariance Approx %
H_T_H_bar = V_bar*(S_bar)^2*(transpose(V_bar));
H_T_H = V*(S)^2*(transpose(V));

C_s_est = (1 / m)*(H_T_H_bar) + var_noise*eye(N);

U3 = [V1_bar, V2_bar];
S3 = blkdiag((S1)^2 - N*var_noise*eye(k), N*var_noise*eye(N - k));
V3 = U3';

%% Column Space %%
range = colspace(sym(V1_bar));

%% H_hat Estimate %%
H_hat = V1_bar*(S1_bar^2 + m*var_noise*eye(k))*V1_bar';
%H_ls = U1*S1*V1';

%% Extract s_hat %%
shat = zeros(N,1);
for i=1:N
    shat(i) = mean(diag(fliplr(H_hat),n-i));
end

%% LPC FINALLY %%
a = lpc(shat, 30);
est_signal = filter([0 -a(2:end)], 1, shat);
X = abs(fft(shat));
L = length(X);
f = shat*(0:floor(L/2))/L;
X2 = X(1:floor(L/2) + 1);

%semilogx(shat)
plot(1:N, shat, 1:N, est_signal, "--")
grid
legend('Original Signal', 'LPC Estimate')