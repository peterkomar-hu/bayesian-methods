data {
	int<lower=0> Ntot;
	int<lower=1> G;
	int<lower=1,upper=G> labels[Ntot];
	real x[Ntot];
}

transformed data {
	int N[G];
	real sumx[G];
	real sumx2[G];
	real m[G];
	real s2[G];

	for (g in 1:G){
		N[g] = 0;
		sumx[g] = 0;
		sumx2[g] = 0;
	}

	for (i in 1:Ntot){
		N[labels[i]] += 1;
		sumx[labels[i]] += x[i];
		sumx2[labels[i]] += x[i]^2;
	}

	for (g in 1:G){
		m[g] = sumx[g] / N[g];
		s2[g] = sumx2[g] / N[g] - m[g]^2;
	}
}

parameters {
	real mu0;
	real<lower=0> sigma0;
	real<lower=0, upper=sigma0> sigma;
}


transformed parameters {
	real z = (sigma / sigma0)^2;
}

model {
	sigma ~ lognormal(0, 4.6);
	sigma0 ~ lognormal(0, 6.9);
	mu0 ~ cauchy(0, 10);
	target += - G * log(sigma0);
	for (g in 1:G){
		target += - 0.5 * log(z + N[g]);
		target += - (N[g] - 1) * log(sigma);
		target += - N[g] / (2 * sigma^2) * ( z/(z+N[g]) * (mu0 - m[g])^2 + s2[g] );
	}
}

