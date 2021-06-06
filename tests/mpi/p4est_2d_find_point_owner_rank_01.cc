// ---------------------------------------------------------------------
//
// Copyright (C) 2009 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------



// one global refinement of a 2d square with p4est

#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>

#include "../tests.h"



template <int dim>
void
test(const Point<dim> &point)
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << " Building hyper_L..." << std::endl;

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_L(triangulation);
  triangulation.refine_global(1);

  deallog << "   Number of cells = " << triangulation.n_global_active_cells()
          << std::endl;
  deallog << "   Number of levels = " << triangulation.n_levels() << std::endl;
  deallog << "   Number of global levels = " << triangulation.n_global_levels()
          << std::endl;
  const unsigned int checksum = triangulation.get_checksum();
  deallog << "   Triangulation checksum = " << checksum << std::endl;
  deallog << "   point = " << point << std::endl;

  ////////////////////////////////////////////////////////////
  // test stuff
  unsigned int point_owner_rank = triangulation.find_point_owner_rank(point);
  deallog << "   rank of point owner = " << point_owner_rank << std::endl;
  ////////////////////////////////////////////////////////////

  deallog << "--- Reached end of test ---" << std::endl;
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    log;


  /*
   * We craete a distrubuted mesh with three cells (tree roots), e.g., a
   * hyper_L. We want to find the mpi rank of a single fixed point. On all
   * processes we must find the same owner.
   */
  const Point<2> point(0.25, 0.75);


  initlog();

  deallog.push("2D-Test");
  test<2>(point);
  deallog.pop();
}
