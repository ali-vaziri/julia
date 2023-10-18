SIZE = 4095

using ImplicitGlobalGrid
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

@parallel function compute_V!(Vx::Data.Array, Vy::Data.Array, P::Data.Array, dt::Data.Number, ρ::Data.Number, dx::Data.Number, dy::Data.Number)
    @inn(Vx) = @inn(Vx) - dt/ρ*@d_xi(P)/dx
    @inn(Vy) = @inn(Vy) - dt/ρ*@d_yi(P)/dy
    return
end

@parallel function compute_P!(P::Data.Array, Vx::Data.Array, Vy::Data.Array, dt::Data.Number, k::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(P) = @all(P) - dt*k*(@d_xa(Vx)/dx + @d_ya(Vy)/dy)
    return
end

##################################################
@views function acoustic2D()
    # Physics
    lx, ly    = 40.0, 40.0  # domain extends
    k         = 1.0         # bulk modulus
    ρ         = 1.0         # density
    # Numerics
    nx, ny    = SIZE, SIZE    # numerical grid resolution; should be a mulitple of 32-1 for optimal GPU perf
    me, dims,nprocs   = init_global_grid(nx, ny, 1)
    global wtime0 = Base.time()
    nt        = 200        # number of timesteps
    # Derived numerics
    dx, dy    = lx/(nx-1), ly/(ny-1) # cell sizes
    # Array allocations
    P         = @zeros(nx  ,ny  )
    Vx        = @zeros(nx+1,ny  )
    Vy        = @zeros(nx  ,ny+1)
    # Initial conditions
    P        .= [exp(-((ix-1)*dx-0.5*lx)^2 -((iy-1)*dy-0.5*ly)^2) for ix=1:size(P,1), iy=1:size(P,2)]
    dt        = min(dx,dy)/sqrt(k/ρ)/4.1
    # Time loop
    for it = 1:nt
        @parallel compute_V!(Vx, Vy, P, dt, ρ, dx, dy)
        @parallel compute_P!(P, Vx, Vy, dt, k, dx, dy)
        update_halo!(P)
    end
    wtime = Base.time() - wtime0
    println("time (s) = $wtime") 
    finalize_global_grid();
end

acoustic2D()
