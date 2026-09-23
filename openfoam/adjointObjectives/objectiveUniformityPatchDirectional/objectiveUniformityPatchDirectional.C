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

\*---------------------------------------------------------------------------*/

#include "objectiveUniformityPatchDirectional.H"
#include "createZeroField.H"
#include "coupledFvPatch.H"
#include "HashSet.H"
#include "IOmanip.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

namespace objectives
{

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

defineTypeNameAndDebug(objectiveUniformityPatchDirectional, 0);
addToRunTimeSelectionTable
(
    objectiveIncompressible,
    objectiveUniformityPatchDirectional,
    dictionary
);


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

Foam::labelList
objectiveUniformityPatchDirectional::selectPatches(const word& key) const
{
    // Prefer the specific key, fall back to the shared "patches" key
    wordRes patchSelection;
    if
    (
        dict().readIfPresent(key, patchSelection)
     || dict().readIfPresent("patches", patchSelection)
    )
    {
        return mesh_.boundaryMesh().patchSet(patchSelection).sortedToc();
    }

    // Otherwise, pick patches up based on the mass flow.
    // Note: a non-zero U initialisation should be used in order to pick up the
    // outlet patches correctly
    WarningInFunction
        << "No patches provided to " << type() << " for key '" << key
        << "'. Choosing them according to the patch mass flows" << nl;

    DynamicList<label> reportPatches(mesh_.boundary().size());
    const surfaceScalarField& phi = vars_.phiInst();
    forAll(mesh_.boundary(), patchI)
    {
        const fvsPatchScalarField& phiPatch = phi.boundaryField()[patchI];
        if (!isA<coupledFvPatch>(mesh_.boundary()[patchI]))
        {
            const scalar mass = gSum(phiPatch);
            if (mass > SMALL)
            {
                reportPatches.push_back(patchI);
            }
        }
    }

    labelList result;
    result.transfer(reportPatches);
    return result;
}


void objectiveUniformityPatchDirectional::initialize()
{
    parallelPatches_ = selectPatches("parallelPatches");
    transversePatches_ = selectPatches("transversePatches");

    if (parallelPatches_.empty() && transversePatches_.empty())
    {
        FatalErrorInFunction
            << "No valid patch name on which to minimize " << type() << endl
            << exit(FatalError);
    }

    // Union of both patch sets + membership flags
    labelHashSet parSet(parallelPatches_);
    labelHashSet transSet(transversePatches_);

    labelHashSet unionSet(parallelPatches_);
    unionSet.insert(transversePatches_);
    unionPatches_ = unionSet.sortedToc();

    parInUnion_.setSize(unionPatches_.size());
    transInUnion_.setSize(unionPatches_.size());
    forAll(unionPatches_, i)
    {
        parInUnion_[i] = parSet.found(unionPatches_[i]);
        transInUnion_[i] = transSet.found(unionPatches_[i]);
    }

    if (debug)
    {
        Info<< "Minimizing " << type() << nl
            << "    uniform direction " << direction_
            << ", transverse weight " << transverseWeight_ << nl
            << "    parallel patches:";
        for (const label patchI : parallelPatches_)
        {
            Info<< " " << mesh_.boundary()[patchI].name();
        }
        Info<< nl << "    transverse patches:";
        for (const label patchI : transversePatches_)
        {
            Info<< " " << mesh_.boundary()[patchI].name();
        }
        Info<< endl;
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

objectiveUniformityPatchDirectional::objectiveUniformityPatchDirectional
(
    const fvMesh& mesh,
    const dictionary& dict,
    const word& adjointSolverName,
    const word& primalSolverName
)
:
    objectiveIncompressible(mesh, dict, adjointSolverName, primalSolverName),
    direction_(dict.getOrDefault<vector>("uniformDirection", vector(1, 0, 0))),
    transverseWeight_(dict.getOrDefault<scalar>("transverseWeight", 1.0)),
    parallelPatches_(),
    transversePatches_(),
    unionPatches_(),
    parInUnion_(),
    transInUnion_(),
    sumMagSfPar_(Zero),
    sumMagSfTrans_(Zero),
    UParMean_(Zero),
    UParVar_(Zero),
    UTransMeanSq_(Zero)
{
    // Normalise the uniform direction
    const scalar magDir = mag(direction_);
    if (magDir < SMALL)
    {
        FatalErrorInFunction
            << "uniformDirection has (near) zero magnitude: " << direction_
            << exit(FatalError);
    }
    direction_ /= magDir;

    // Resolve patch sets
    initialize();

    // Allocate boundary field pointers
    bdJdvPtr_.reset(createZeroBoundaryPtr<vector>(mesh_));
    bdJdvnPtr_.reset(createZeroBoundaryPtr<scalar>(mesh_));
    bdJdvtPtr_.reset(createZeroBoundaryPtr<vector>(mesh_));
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

scalar objectiveUniformityPatchDirectional::J()
{
    J_ = Zero;

    const volVectorField& U = vars_.UInst();

    // Parallel term: variance of U.direction_ over the parallel patch set
    sumMagSfPar_ = Zero;
    scalar sumUd = Zero;
    forAll(parallelPatches_, oI)
    {
        const label patchI = parallelPatches_[oI];
        const scalarField& magSf = mesh_.boundary()[patchI].magSf();
        const fvPatchVectorField& Ub = U.boundaryField()[patchI];
        sumMagSfPar_ += gSum(magSf);
        sumUd += gSum((Ub & direction_)*magSf);
    }

    UParVar_ = Zero;
    if (sumMagSfPar_ > SMALL)
    {
        UParMean_ = sumUd/sumMagSfPar_;
        forAll(parallelPatches_, oI)
        {
            const label patchI = parallelPatches_[oI];
            const scalarField& magSf = mesh_.boundary()[patchI].magSf();
            const fvPatchVectorField& Ub = U.boundaryField()[patchI];
            const scalarField p((Ub & direction_) - UParMean_);
            UParVar_ += gSum(sqr(p)*magSf);
        }
        UParVar_ /= sumMagSfPar_;
    }

    // Transverse term: mean-square of U - (U.direction_) direction_
    sumMagSfTrans_ = Zero;
    UTransMeanSq_ = Zero;
    forAll(transversePatches_, oI)
    {
        const label patchI = transversePatches_[oI];
        const scalarField& magSf = mesh_.boundary()[patchI].magSf();
        const fvPatchVectorField& Ub = U.boundaryField()[patchI];
        sumMagSfTrans_ += gSum(magSf);
        const vectorField Ut(Ub - (Ub & direction_)*direction_);
        UTransMeanSq_ += gSum(magSqr(Ut)*magSf);
    }
    if (sumMagSfTrans_ > SMALL)
    {
        UTransMeanSq_ /= sumMagSfTrans_;
    }

    J_ = 0.5*UParVar_ + 0.5*transverseWeight_*UTransMeanSq_;

    return J_;
}


void objectiveUniformityPatchDirectional::update_boundarydJdv()
{
    const volVectorField& U = vars_.U();

    forAll(unionPatches_, i)
    {
        const label patchI = unionPatches_[i];
        const fvPatchVectorField& Ub = U.boundaryField()[patchI];

        vectorField contrib(Ub.size(), Zero);
        if (parInUnion_[i] && sumMagSfPar_ > SMALL)
        {
            const scalarField p((Ub & direction_) - UParMean_);
            contrib += p*direction_/sumMagSfPar_;
        }
        if (transInUnion_[i] && sumMagSfTrans_ > SMALL)
        {
            const vectorField Ut(Ub - (Ub & direction_)*direction_);
            contrib += transverseWeight_*Ut/sumMagSfTrans_;
        }

        bdJdvPtr_()[patchI] = contrib;
    }
}


void objectiveUniformityPatchDirectional::update_boundarydJdvn()
{
    const volVectorField& U = vars_.U();

    forAll(unionPatches_, i)
    {
        const label patchI = unionPatches_[i];
        const fvPatchVectorField& Ub = U.boundaryField()[patchI];
        const tmp<vectorField> nf = mesh_.boundary()[patchI].nf();

        vectorField contrib(Ub.size(), Zero);
        if (parInUnion_[i] && sumMagSfPar_ > SMALL)
        {
            const scalarField p((Ub & direction_) - UParMean_);
            contrib += p*direction_/sumMagSfPar_;
        }
        if (transInUnion_[i] && sumMagSfTrans_ > SMALL)
        {
            const vectorField Ut(Ub - (Ub & direction_)*direction_);
            contrib += transverseWeight_*Ut/sumMagSfTrans_;
        }

        bdJdvnPtr_()[patchI] = (contrib & nf);
    }
}


void objectiveUniformityPatchDirectional::update_boundarydJdvt()
{
    const volVectorField& U = vars_.U();

    forAll(unionPatches_, i)
    {
        const label patchI = unionPatches_[i];
        const fvPatchVectorField& Ub = U.boundaryField()[patchI];
        const tmp<vectorField> nf = mesh_.boundary()[patchI].nf();

        vectorField contrib(Ub.size(), Zero);
        if (parInUnion_[i] && sumMagSfPar_ > SMALL)
        {
            const scalarField p((Ub & direction_) - UParMean_);
            contrib += p*direction_/sumMagSfPar_;
        }
        if (transInUnion_[i] && sumMagSfTrans_ > SMALL)
        {
            const vectorField Ut(Ub - (Ub & direction_)*direction_);
            contrib += transverseWeight_*Ut/sumMagSfTrans_;
        }

        bdJdvtPtr_()[patchI] = (contrib - (contrib & nf())*nf());
    }
}


void objectiveUniformityPatchDirectional::addHeaderColumns() const
{
    OFstream& file = objFunctionFilePtr_();
    file<< setw(width_) << "UParMean" << " ";
    file<< setw(width_) << "UParVar" << " ";
    file<< setw(width_) << "UTransMeanSq" << " ";
}


void objectiveUniformityPatchDirectional::addColumnValues() const
{
    OFstream& file = objFunctionFilePtr_();
    file<< setw(width_) << UParMean_ << " ";
    file<< setw(width_) << UParVar_ << " ";
    file<< setw(width_) << UTransMeanSq_ << " ";
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace objectives
} // End namespace Foam

// ************************************************************************* //
