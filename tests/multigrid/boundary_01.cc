//----------------------------------------------------------------------------
//    $Id$
//    Version: $Name$
//
//    Copyright (C) 2006 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------------------------------------------------------

// check MGTools::make_boundary_list

#include "../tests.h"
#include <base/logstream.h>
#include <base/function.h>
#include <lac/vector.h>
#include <lac/block_vector.h>
#include <grid/tria.h>
#include <grid/tria_iterator.h>
#include <grid/tria_accessor.h>
#include <grid/grid_generator.h>
#include <dofs/function_map.h>
#include <fe/fe_dgq.h>
#include <fe/fe_q.h>
#include <fe/fe_raviart_thomas.h>
#include <fe/fe_system.h>
#include <multigrid/mg_dof_accessor.h>
#include <multigrid/mg_dof_handler.h>
#include <multigrid/mg_tools.h>

#include <fstream>
#include <iomanip>
#include <iomanip>
#include <algorithm>

using namespace std;

void log_vector (const std::vector<std::set<unsigned int> >& count)
{
  for (unsigned int l=0;l<count.size();++l)
    {
      deallog << "Level " << l << ':';
      for (std::set<unsigned int>::const_iterator c=count[l].begin();
	   c != count[l].end();++c)
	deallog << ' ' << *c;
      deallog << std::endl;
    }
}


template <int dim>
void check_fe(FiniteElement<dim>& fe)
{
  deallog << fe.get_name() << std::endl;
  
  Triangulation<dim> tr;
  GridGenerator::hyper_cube(tr);
  tr.refine_global(2);
  ZeroFunction<dim> zero;
  typename FunctionMap<dim>::type fmap;
  fmap.insert(std::make_pair(0, &zero));
  
  MGDoFHandler<dim> mgdof(tr);
  mgdof.distribute_dofs(fe);
  
  std::vector<std::set<unsigned int> > boundary_indices(tr.n_levels());
  MGTools::make_boundary_list(mgdof, fmap, boundary_indices);
  log_vector(boundary_indices);
}


template <int dim>
void check()
{
  FE_Q<dim> q1(1);
  FE_Q<dim> q2(2);
//  FE_DGQ<dim> dq1(1);
  
  FESystem<dim> s1(q1, 2, q2,1);

  check_fe(q1);
  check_fe(q2);
  check_fe(s1);
}

int main()
{
  std::ofstream logfile("boundary_01/output");
  deallog << std::setprecision(3);
  deallog.attach(logfile);
  deallog.depth_console(0);
  deallog.threshold_double(1.e-10);

  check<2> ();
  check<3> ();
}
