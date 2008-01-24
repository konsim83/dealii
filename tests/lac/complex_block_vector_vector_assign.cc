//--------------------------------------------------------------------
//    $Id$
//    Version: $Name$ 
//
//    Copyright (C) 2006, 2007, 2008 by the deal.II authors
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//--------------------------------------------------------------------

// check assignment between block vectors and regular vectors

#include "../tests.h"
#include <base/logstream.h>
#include <lac/block_vector.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <utility>
#include <cmath>

template <typename Vector1, typename Vector2>
bool operator == (const Vector1 &v1,
		  const Vector2 &v2)
{
  if (v1.size() != v2.size())
    return false;
  for (unsigned int i=0; i<v1.size(); ++i)
    if (v1(i) != v2(i))
      return false;
  return true;
}


void test ()
{
  std::vector<unsigned int> ivector(4);
  ivector[0] = 2;
  ivector[1] = 4;
  ivector[2] = 3;
  ivector[3] = 5;
  
  BlockVector<std::complex<double> > v1(ivector);
  Vector<std::complex<double> > v2(v1.size());

  for (unsigned int i=0; i<v1.size(); ++i)
    v1(i) = 1+i*i;

  v2 = v1;
  Assert (v1==v2, ExcInternalError());

  BlockVector<std::complex<double> > v3 (ivector);
  v3 = v2;
  Assert (v3==v2, ExcInternalError());
  Assert (v3==v1, ExcInternalError());

  deallog << "OK" << std::endl;
}





int main ()
{
  std::ofstream logfile("complex_block_vector_vector_assign/output");
  logfile.setf(std::ios::fixed);
  deallog << std::setprecision(3);
  deallog.attach(logfile);
  deallog.depth_console(0);
  deallog.threshold_double(1.e-10);
  
  try
    {
      test ();
    }
  catch (std::exception &e)
    {
      deallog << std::endl << std::endl
	   << "----------------------------------------------------"
	   << std::endl;
      deallog << "Exception on processing: " << e.what() << std::endl
	   << "Aborting!" << std::endl
	   << "----------------------------------------------------"
	   << std::endl;
				       // abort
      return 2;
    }
  catch (...) 
    {
      deallog << std::endl << std::endl
	   << "----------------------------------------------------"
	   << std::endl;
      deallog << "Unknown exception!" << std::endl
	   << "Aborting!" << std::endl
	   << "----------------------------------------------------"
	   << std::endl;
				       // abort
      return 3;
    };
  
  
  return 0;
}

