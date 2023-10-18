SIZE = 200

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@init_parallel_stencil(Threads, Float64, 3)

@parallel function compute_q!(qx::Data.Array, qy::Data.Array, qz::Data.Array, T::Data.Array,
    lam::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(qx)    = -lam * (@d_xi(T) / dx);                                                           # Fourier's law of heat conduction: q_x   = -λ δT/δx
    @all(qy)    = -lam * (@d_yi(T) / dy);                                                           # ...                               q_y   = -λ δT/δy
    @all(qz)    = -lam * (@d_zi(T) / dz); 
    return
end

@parallel function compute_t!(qx::Data.Array, qy::Data.Array, qz::Data.Array,  
    Cp::Data.Array, T::Data.Array, dTedt::Data.Array,
    dt::Data.Number, dx::Data.Number, dy::Data.Number, dz::Data.Number)
    @all(dTedt) = (1.0 /@inn(Cp)) * (-(@d_xa(qx)/dx) - (@d_ya(qy)/dy) - (@d_za(qz)/dz));               # Conservation of energy:           δT/δt = 1/cₚ (-δq_x/δx - δq_y/dy - δq_z/dz)
    @inn(T) = @inn(T) + (dt * @all(dTedt));                                                            # Update of temperature             T_new = T_old + δT/δt
    return
end

@views function diffusion3D()

    # Numerics
    nx, ny, nz = SIZE, SIZE, SIZE;                             # Number of gridpoints in dimensions x, y and z
    nt         = 100;                                       # Number of time steps
    me, dims,nprocs   = init_global_grid(nx, ny, nz)
    global wtime0 = Base.time()
    # Physics
    lam        = 1.0;                                       # Thermal conductivity
    cp_min     = 1.0;                                       # Minimal heat capacity
    lx, ly, lz = 10.0, 10.0, 10.0;                          # Length of computational domain in dimension x, y and z
    dx         = lx/(nx_g()-1);                             # Space step in dimension x
    dy         = ly/(ny_g()-1);                             # ...        in dimension y
    dz         = lz/(nz_g()-1);                             # ...        in dimension z
    # Array initializations
    T     = @zeros(nx,   ny,   nz  );
    Cp    = @zeros(nx,   ny,   nz  );
    dTedt = @zeros(nx-2, ny-2, nz-2);
    qx    = @zeros(nx-1, ny-2, nz-2);
    qy    = @zeros(nx-2, ny-1, nz-2);
    qz    = @zeros(nx-2, ny-2, nz-1);

    # Initial conditions (heat capacity and temperature with two Gaussian anomalies each)
    Cp .= cp_min .+ [5*exp(-((x_g(ix,dx,Cp)-lx/1.5))^2-((y_g(iy,dy,Cp)-ly/2))^2-((z_g(iz,dz,Cp)-lz/1.5))^2) +
                     5*exp(-((x_g(ix,dx,Cp)-lx/3.0))^2-((y_g(iy,dy,Cp)-ly/2))^2-((z_g(iz,dz,Cp)-lz/1.5))^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]
    T  .= [100*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/3.0)/2)^2) +
            50*exp(-((x_g(ix,dx,T)-lx/2)/2)^2-((y_g(iy,dy,T)-ly/2)/2)^2-((z_g(iz,dz,T)-lz/1.5)/2)^2) for ix=1:size(T,1), iy=1:size(T,2), iz=1:size(T,3)]

    # Time loop
    dt = min(dx*dx,dy*dy,dz*dz)*cp_min/lam/8.1;                                               # Time step for the 3D Heat diffusion
    for it = 1:nt
        @parallel compute_q!(qx, qy, qz, T, lam, dx, dy, dz) 
        @parallel compute_t!(qx, qy, qz, Cp, T, dTedt, dt, dx, dy, dz)                       # ...                               q_z   = -λ δT/δz
        update_halo!(T)                                                                      # Update the halo of T
    end
    wtime = Base.time() - wtime0
    println("time (s) = $wtime")         
    finalize_global_grid();                                                                  # Finalize the implicit global grid
end

diffusion3D()
