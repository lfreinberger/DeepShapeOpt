/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2007-2022 PCOpt/NTUA
    Copyright (C) 2013-2022 FOSS GP
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.


Class
    Foam::objectives::objectiveUniformityPatchDirectional

Description
    Directional variant of objectiveUniformityPatch.

    Makes the velocity uniform along a single direction d (default x) on the
    "parallel" patch set, while driving the velocity components transverse to d
    to zero on the (possibly different) "transverse" patch set:

        J = 0.5*Var_par(U.d)
          + 0.5*w*meanSq_trans(U - (U.d) d)

    with each term area-normalised over its own patch set and w a configurable
    transverse weight. When parallelPatches == transversePatches, w = 1 and
    d = (1 0 0), this reduces to objectiveUniformityPatch with the target mean
    velocity (UMean_x, 0, 0).

    Dictionary entries:
        uniformDirection   (1 0 0);   // direction made uniform (normalised)
        transverseWeight   1.0;        // weight on the transverse (->0) term
        parallelPatches    (...);      // where U.d is made uniform
        transversePatches  (...);      // where transverse U is driven to 0

    Either patch key falls back to the single "patches" key when absent, so the
    symmetric single-patch case behaves like objectiveUniformityPatch.

SourceFiles
    objectiveUniformityPatchDirectional.C

\*---------------------------------------------------------------------------*/

#ifndef objectiveUniformityPatchDirectional_H
#define objectiveUniformityPatchDirectional_H

#include "objectiveIncompressible.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

namespace objectives
{

/*---------------------------------------------------------------------------*\
              Class objectiveUniformityPatchDirectional Declaration
\*---------------------------------------------------------------------------*/

class objectiveUniformityPatchDirectional
:
    public objectiveIncompressible
{
    // Private data

        //- Unit direction made uniform (transverse components driven to 0)
        vector direction_;

        //- Weight on the transverse (drive-to-zero) term
        scalar transverseWeight_;

        //- Patches where U.direction_ is made uniform
        labelList parallelPatches_;

        //- Patches where the transverse velocity is driven to zero
        labelList transversePatches_;

        //- Union of the two patch sets (boundary terms are written here)
        labelList unionPatches_;

        //- Membership flags aligned with unionPatches_
        boolList parInUnion_;
        boolList transInUnion_;

        //- Cached area sums per set (set in J(), reused by the dJdv updates)
        scalar sumMagSfPar_;
        scalar sumMagSfTrans_;

        //- Mean of U.direction_ over the parallel patch set (set in J())
        scalar UParMean_;

        //- Diagnostics
        scalar UParVar_;
        scalar UTransMeanSq_;


    // Private Member Functions

        //- Resolve the objective patch sets and build the union/membership data
        void initialize();

        //- Patch labels for a dictionary key, falling back to "patches"
        labelList selectPatches(const word& key) const;


public:

    //- Runtime type information
    TypeName("uniformityPatchDirectional");


    // Constructors

        //- from components
        objectiveUniformityPatchDirectional
        (
            const fvMesh& mesh,
            const dictionary& dict,
            const word& adjointSolverName,
            const word& primalSolverName
        );


    //- Destructor
    virtual ~objectiveUniformityPatchDirectional() = default;


    // Member Functions

        //- Return the objective function value
        virtual scalar J();

        //- Update values to be added to the adjoint outlet velocity
        virtual void update_boundarydJdv();

        //- Update values to be added to the adjoint outlet pressure
        virtual void update_boundarydJdvn();

        //- Update values to be added to the adjoint outlet tangential velocity
        virtual void update_boundarydJdvt();

        // Helper write functions

            //- Write headers for additional columns
            virtual void addHeaderColumns() const;

            //- Write information to additional columns
            virtual void addColumnValues() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace objectives
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
